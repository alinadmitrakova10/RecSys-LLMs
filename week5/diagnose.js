const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.setDefaultTimeout(1800000);
  
  page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message));
  
  await page.goto('http://localhost:8000/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(5000);
  
  // Initialize model first
  await page.evaluate(() => {
    initializeFactors(trainSet, NUM_FACTORS);
  });
  await page.waitForTimeout(2000);
  
  // Get initial factor matrices stats before training
  const initStats = await page.evaluate(() => {
    const u = model.userFactors.arraySync();
    const i = model.itemFactors.arraySync();
    const uNorm = Math.sqrt(u.flat().reduce((sum, v) => sum + v*v, 0));
    const iNorm = Math.sqrt(i.flat().reduce((sum, v) => sum + v*v, 0));
    return {
      userFactorsSample: u[1].slice(0, 5),
      itemFactorsSample: i[1].slice(0, 5),
      userL2Norm: uNorm,
      itemL2Norm: iNorm,
    };
  });
  console.log('=== INITIAL FACTORS (before train) ===');
  console.log(initStats);
  
  // Check hyperparameters types
  const hpTypes = await page.evaluate(() => {
    return {
      epochs: { value: 20, type: typeof 20 },
      lr: { value: 0.01, type: typeof 0.01 },
      reg: { value: 0.05, type: typeof 0.05 },
      batchSize: { value: 64, type: typeof 64 },
    };
  });
  console.log('=== HYPERPARAMETER TYPES ===');
  console.log(hpTypes);
  
  // Train for just 1 epoch to check first epoch
  console.log('=== TRAINING 1 EPOCH ===');
  await page.evaluate(() => {
    const opts = { epochs: 1, lr: 0.01, reg: 0.05, batchSize: 64 };
    trainingHistory = trainMF(trainSet, valSet, opts);
  });
  
  await page.waitForTimeout(30000);
  
  const afterEpoch1 = await page.evaluate(() => {
    const u = model.userFactors.arraySync();
    const i = model.itemFactors.arraySync();
    const uNorm = Math.sqrt(u.flat().reduce((sum, v) => sum + v*v, 0));
    const iNorm = Math.sqrt(i.flat().reduce((sum, v) => sum + v*v, 0));
    return {
      userFactorsSample: u[1].slice(0, 5),
      itemFactorsSample: i[1].slice(0, 5),
      userL2Norm: uNorm,
      itemL2Norm: iNorm,
      history: trainingHistory,
    };
  });
  console.log('=== AFTER EPOCH 1 ===');
  console.log(afterEpoch1);
  
  // Train full 20 epochs (fresh init)
  console.log('=== TRAINING FULL 20 EPOCHS ===');
  await page.evaluate(() => {
    const opts = { epochs: 20, lr: 0.01, reg: 0.05, batchSize: 64 };
    initializeFactors(trainSet, NUM_FACTORS);
    trainingHistory = trainMF(trainSet, valSet, opts);
  });
  
  await page.waitForTimeout(300000);
  
  const afterFull = await page.evaluate(() => {
    const u = model.userFactors.arraySync();
    const i = model.itemFactors.arraySync();
    const uNorm = Math.sqrt(u.flat().reduce((sum, v) => sum + v*v, 0));
    const iNorm = Math.sqrt(i.flat().reduce((sum, v) => sum + v*v, 0));
    return {
      userFactorsSample: u[1].slice(0, 5),
      itemFactorsSample: i[1].slice(0, 5),
      userL2Norm: uNorm,
      itemL2Norm: iNorm,
      history: trainingHistory,
    };
  });
  console.log('=== AFTER 20 EPOCHS ===');
  console.log(afterFull);
  
  // Check memory
  const memory = await page.evaluate(() => tf.memory());
  console.log('=== TF MEMORY ===');
  console.log(memory);
  
  // Verify batch count
  const batchCheck = await page.evaluate(() => {
    const n = trainSet.length;
    const batchSize = 64;
    return { n, batchSize, expectedBatches: Math.ceil(n / batchSize) };
  });
  console.log('=== BATCH COUNT ===');
  console.log(batchCheck);
  
  // Check optimizer variables
  const optCheck = await page.evaluate(() => {
    initializeFactors(trainSet, NUM_FACTORS);
    return {
      userFactorsIsVar: model.userFactors instanceof tf.Variable,
      itemFactorsIsVar: model.itemFactors instanceof tf.Variable,
      userFactorsShape: model.userFactors.shape,
      itemFactorsShape: model.itemFactors.shape,
    };
  });
  console.log('=== OPTIMIZER VARIABLES CHECK ===');
  console.log(optCheck);
  
  await browser.close();
})();