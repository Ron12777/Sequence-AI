const {defineConfig} = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/ui',
  timeout: 60000,
  expect: {timeout: 15000},
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: !!process.env.CI,
  reporter: [['list'], ['html', {open: 'never'}]],
  use: {
    baseURL: 'http://127.0.0.1:8766',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {name: 'desktop', use: {browserName: 'chromium', viewport: {width: 1440, height: 1000}}},
    {name: 'mobile', use: {browserName: 'chromium', viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true}},
    {name: 'webkit', use: {browserName: 'webkit', viewport: {width: 1440, height: 1000}}},
    {name: 'mobile-webkit', use: {browserName: 'webkit', viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true}},
  ],
  webServer: {
    command: 'python3 -m http.server 8766 --bind 127.0.0.1 --directory web/static',
    url: 'http://127.0.0.1:8766',
    reuseExistingServer: false,
  },
});
