/**
 * Batched PPR with reusable TensorFlow.js transitions. Each row is one user's
 * probability vector: p_row * Q equals (Q^T * p_column)^T. MovieLens uses only
 * the normalized R and R^T blocks supplied by rowNormalize, never square Q.
 */
const HW6_BATCH_PPR = (function () {
  /** Caller owns the returned engine and must dispose it, including on failure. */
  async function create(graph, options = {}) {
    await tf.ready();
    const transition = options.transition;
    const n = transition ? graphValidateTransition(transition) : graph && graph.nodeCount;
    const capacity = options.maxBatchSize || 16;
    if (!Number.isInteger(n) || n < 1 || !Number.isInteger(capacity) || capacity < 1 || capacity > 64) {
      throw new Error("Batch PPR: require a nonempty graph and capacity in [1,64]");
    }
    let blocks = null;
    let q = null;
    let mask = null;
    let disposed = false;
    let busy = false;
    try {
      if (transition) q = tf.tensor2d(transition.Q, [n, n]);
      else blocks = rowNormalize(graph);
      mask = tf.tensor2d(Array.from(transition ? transition.danglingMask : blocks.danglingMask, x => x ? 1 : 0), [1, n]);
    } catch (error) {
      if (blocks) blocks.dispose();
      tf.dispose([q, mask].filter(Boolean));
      throw error;
    }

    /**
     * Independent seeds, L1 residuals and first-converged snapshots. Freeze a row
     * at its own tolerance crossing, not at the last row's convergence. Trace is
     * diagnostic only. All results contain JS data; owned run tensors are freed.
     */
    async function runNodes(seedNodes, alpha, opts = {}, userIds = null) {
      if (disposed || busy) throw new Error("Batch PPR: engine is disposed or already running");
      if (!Array.isArray(seedNodes) || !seedNodes.length || seedNodes.length > capacity ||
          new Set(seedNodes).size !== seedNodes.length) throw new Error("Batch PPR: invalid seed batch");
      const b = seedNodes.length;
      const tol = opts.tol === undefined ? 1e-6 : opts.tol;
      const maxIter = opts.maxIter === undefined ? 500 : opts.maxIter;
      if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1 || !Number.isFinite(tol) || tol <= 0 ||
          !Number.isInteger(maxIter) || maxIter < 1 || maxIter > 500) throw new Error("Batch PPR: invalid alpha/tolerance/cap");
      if (seedNodes.some(node => !Number.isInteger(node) || node < 0 || node >= n || (!transition && node >= graph.nUsers))) {
        throw new Error("Batch PPR: invalid or non-user restart node index");
      }
      busy = true;
      const started = performance.now();
      let v = null;
      let p = null;
      let pending = null;
      const active = new Uint8Array(b).fill(1);
      const iterations = new Int32Array(b);
      const residuals = new Float64Array(b).fill(Infinity);
      const traces = Array.from({ length: b }, () => []);
      let remaining = b;
      let batchIterations = 0;
      try {
        const restart = new Float32Array(b * n);
        seedNodes.forEach((node, row) => { restart[row * n + node] = 1; });
        v = tf.tensor2d(restart, [b, n]);
        p = v.clone();
        for (let iteration = 1; iteration <= maxIter; iteration++) {
          pending = tf.tidy(() => {
            let linked;
            if (q) linked = tf.matMul(p, q);
            else {
              const pu = p.slice([0, 0], [b, blocks.nUsers]);
              const pm = p.slice([0, blocks.nUsers], [b, blocks.nMovies]);
              linked = tf.concat([tf.matMul(pm, blocks.movieToUser), tf.matMul(pu, blocks.userToMovie)], 1);
            }
            const restartMass = p.mul(mask).sum(1).reshape([b, 1]).mul(alpha).add(1 - alpha);
            const proposed = linked.mul(alpha).add(v.mul(restartMass));
            const delta = proposed.sub(p).abs().sum(1);
            const activeMask = tf.tensor2d(active, [b, 1], "bool").tile([1, n]);
            const next = tf.where(activeMask, proposed, p);
            const stats = opts.trace ? tf.stack([delta, next.sum(1), next.min(1)], 1) : delta;
            return { next, stats };
          });
          const values = await pending.stats.data();
          for (let row = 0; row < b; row++) {
            if (!active[row]) continue;
            const residual = values[opts.trace ? row * 3 : row];
            if (!Number.isFinite(residual)) throw new Error("Batch PPR: non-finite residual");
            iterations[row] = iteration;
            residuals[row] = residual;
            if (opts.trace) traces[row].push({ iteration, residual, mass: values[row * 3 + 1], minimum: values[row * 3 + 2] });
            if (residual < tol) { active[row] = 0; remaining--; }
          }
          p.dispose(); p = pending.next; pending.stats.dispose(); pending = null;
          batchIterations = iteration;
          if (!remaining || iteration === maxIter) break;
          if (iteration % 20 === 0) await tf.nextFrame();
        }
        const values = await p.data();
        const results = seedNodes.map((seedNode, row) => {
          const probabilities = Float64Array.from(values.subarray(row * n, (row + 1) * n));
          const mass = probabilities.reduce((sum, value) => sum + value, 0);
          if (probabilities.some(value => !Number.isFinite(value) || value < 0) || Math.abs(mass - 1) > 1e-5) {
            throw new Error("Batch PPR: invalid final probability distribution");
          }
          return { userId: userIds ? userIds[row] : null, seedNode, p: probabilities,
            iterations: iterations[row], residual: residuals[row], converged: residuals[row] < tol,
            mass, trace: traces[row] };
        });
        return { results, batchIterations, backend: tf.getBackend(), runtimeMs: performance.now() - started };
      } finally {
        if (pending) tf.dispose(pending);
        tf.dispose([v, p].filter(Boolean));
        busy = false;
      }
    }

    async function runUsers(userIds, alpha, opts = {}) {
      if (!graph || transition) throw new Error("Batch PPR: raw user IDs require a MovieLens-style graph");
      const nodes = userIds.map(userId => graph.nodeOfUser(userId));
      return runNodes(nodes, alpha, opts, userIds);
    }

    function dispose() {
      if (busy) throw new Error("Batch PPR: cannot dispose a running engine");
      if (disposed) return;
      if (blocks) blocks.dispose();
      tf.dispose([q, mask].filter(Boolean));
      disposed = true;
    }
    return { runUsers, runNodes, dispose, nodeCount: n, capacity,
      blockShapes: blocks ? [blocks.userToMovie.shape.slice(), blocks.movieToUser.shape.slice()] : null };
  }
  return { create };
})();
