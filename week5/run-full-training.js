const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.setDefaultTimeout(1800000); // 30 min timeout
  
  page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message));
  
  await page.goto('http://localhost:8000/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(5000);
  
  // Get all the static info first
  const datasetSummary = await page.textContent('#dataset-summary-body');
  console.log('=== DATASET SUMMARY ===');
  console.log(datasetSummary);
  
  const splitSummary = await page.textContent('#split-summary-body');
  console.log('=== SPLIT SUMMARY ===');
  console.log(splitSummary);
  
  const tfStatus = await page.textContent('#tf-status');
  console.log('=== TF STATUS ===');
  console.log(tfStatus);
  
  const epochs = await page.inputValue('#epochs');
  const lr = await page.inputValue('#lr');
  const reg = await page.inputValue('#reg');
  const batchSize = await page.inputValue('#batch-size');
  console.log('=== HYPERPARAMETERS ===');
  console.log(`epochs: ${epochs}, lr: ${lr}, reg: ${reg}, batchSize: ${batchSize}`);
  
  // Start training
  console.log('=== STARTING TRAINING ===');
  await page.evaluate(() => {
    document.getElementById('train').click();
  });
  
  // Wait for training curve table to have 20 rows (one per epoch)
  let trainingDone = false;
  for (let i = 0; i < 180; i++) {
    await page.waitForTimeout(10000);
    const trainingCurve = await page.textContent('#training-curve');
    const rowCount = (trainingCurve.match(/<tr>/g) || []).length;
    const status = await page.textContent('#status');
    console.log(`Check ${i+1}: ${rowCount} rows in curve, status: ${status}`);
    if (status.includes('best val RMSE') || (rowCount >= 21 && status.includes('Trained'))) {
      trainingDone = true;
      break;
    }
  }
  
  if (!trainingDone) {
    console.log('Training did not complete within timeout');
  }
  
  const trainingCurve = await page.textContent('#training-curve');
  console.log('=== TRAINING CURVE ===');
  console.log(trainingCurve);
  
  const status = await page.textContent('#status');
  console.log('=== STATUS ===');
  console.log(status);
  
  // Compute test metrics
  const testMetrics = await page.evaluate(() => {
    const baseline = computeBaselineRMSE(testSet);
    const mf = computeRMSEFor(testSet);
    return { baseline, mf };
  });
  console.log('=== TEST METRICS ===');
  console.log(`Baseline test RMSE: ${testMetrics.baseline}`);
  console.log(`MF test RMSE: ${testMetrics.mf}`);
  
  await browser.close();
})();