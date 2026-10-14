const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  // Capture console logs
  page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message));
  
  await page.goto('http://localhost:8000/', { waitUntil: 'networkidle' });
  
  // Wait for the page to fully load and TensorFlow.js to be ready
  await page.waitForTimeout(5000);
  
  // Click Run tests button
  await page.click('#run-tests');
  
  // Wait for tests to complete
  await page.waitForTimeout(5000);
  
  // Get the test log content
  const testLog = await page.textContent('#testLog');
  console.log('=== TEST LOG ===');
  console.log(testLog);
  
  await browser.close();
})();