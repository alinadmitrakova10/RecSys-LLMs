const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.setDefaultTimeout(1800000);
  
  page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message));
  
  await page.goto('http://localhost:8000/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(5000);
  
  // Initialize and check initial factors
  await page.evaluate(() => {
    initializeFactors(trainSet, NUM_FACTORS);
  });
  await page.waitForTimeout(2000);
  
  const initStats = await page.evaluate(() => {
    const u = model.userFactors.arraySync();
    const i = model.itemFactors.arraySync();
    const uNorm = Math.sqrt(u.flat().reduce((sum, v) => sum + v*v, 0));
    const iNorm = Math.sqrt(i.flat().reduce((sum, v) => sum + v*v, 0));
    return { userL2Norm: uNorm, itemL2Norm: iNorm };
  });
  console.log('=== INITIAL FACTORS ===');
  console.log(initStats);
  
  // Train 1 epoch
  await page.evaluate(() => {
    const opts = { epochs: 1, lr: 0.01, reg: 0.05, batchSize: 64 };
    trainingHistory = trainMF(trainSet, valSet, opts);
  });
  await page.waitForTimeout(30000);
  
  const after1 = await page.evaluate(() => {
    const u = model.userFactors.arraySync();
    const i = model.itemFactors.arraySync();
    const uNorm = Math.sqrt(u.flat().reduce((sum, v) => sum + v*v, 0));
    const iNorm = Math.sqrt(i.flat().reduce((sum, v) => sum + v*v, 0));
    return { userL2Norm: uNorm, itemL2Norm: iNorm, history: trainingHistory };
  });
  console.log('=== AFTER EPOCH 1 ===');
  console.log(after1);
  
  // Train full 20 epochs
  await page.evaluate(() => {
    const opts = { epochs: 20, lr: 0.01, reg: 0.05, batchSize: 64 };
    initializeFactors(trainSet, NUM_FACTORS);
    trainingHistory = trainMF(trainSet, valSet, opts);
  });
  await page.waitForTimeout(300000);
  
  const after20 = await page.evaluate(() => {
    const u = model.userFactors.arraySync();
    const i = model.itemFactors.arraySync();
    const uNorm = Math.sqrt(u.flat().reduce((sum, v) => sum + v*v, 0));
    const iNorm = Math.sqrt(i.flat().reduce((sum, v) => sum + v*v, 0));
    return { userL2Norm: uNorm, itemL2Norm: iNorm, history: trainingHistory };
  });
  console.log('=== AFTER 20 EPOCHS (with restore) ===');
  console.log(after20);
  
  // Verify best epoch is 1
  const best = await page.evaluate(() => selectBestCheckpoint(trainingHistory));
  console.log('=== BEST CHECKPOINT ===');
  console.log(best);
  
  await browser.close();
})();