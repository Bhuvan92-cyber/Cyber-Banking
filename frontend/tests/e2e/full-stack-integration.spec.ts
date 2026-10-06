import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('Full-Stack Integration Flow', () => {
  test('should login, upload a CSV, and trigger a report download', async ({ page }) => {
    // 1. Mock session / API login
    await page.route('**/api/auth/login/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: {
          'Set-Cookie': 'sessionid=valid_session_cookie; Path=/; HttpOnly',
        },
        body: JSON.stringify({
          user: { id: 1, username: 'testanalyst', is_staff: false },
        }),
      });
    });

    await page.route('**/api/auth/me/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          user: { id: 1, username: 'testanalyst', is_staff: false },
        }),
      });
    });

    await page.route('**/api/csrf/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        headers: {
          'Set-Cookie': 'csrftoken=mock_csrf_token; Path=/',
        },
        body: JSON.stringify({ csrfToken: 'mock_csrf_token' }),
      });
    });

    // 2. Perform Login via UI
    await page.goto('/login');
    await page.locator('input[type="text"]').fill('testanalyst');
    await page.locator('input[type="password"]').fill('TestPassword123!');
    await page.locator('button[type="submit"]').click();

    // Set cookie on browser context to satisfy Next.js middleware
    await page.context().addCookies([
      {
        name: 'sessionid',
        value: 'valid_session_cookie',
        domain: 'localhost',
        path: '/',
      },
    ]);

    // 3. Navigate to Upload Page
    await page.goto('/dashboard/upload');
    await expect(page.locator('h2')).toContainText('Upload Banking Dataset');

    // 4. Mock upload endpoint
    await page.route('**/api/upload/', async (route) => {
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({
          message: 'Dataset uploaded successfully.',
          dataset_id: 'dummy_data.csv',
          summary: {
            filename: 'dummy_data.csv',
            rows: 5,
            columns: 6,
            column_names: ['Age', 'Balance', 'Transaction_Count', 'Credit_Score', 'Account_Type', 'target'],
            missing_values: {},
            dtypes: {},
            preview: [],
          },
        }),
      });
    });

    // 5. Upload dummy_data.csv
    const filePath = path.join(__dirname, '../fixtures/dummy_data.csv');
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(filePath);

    // 6. Verify summary displays
    await expect(page.locator('text=Dataset Profile:')).toBeVisible({ timeout: 10000 });
    await expect(page.locator('text=Total Rows')).toBeVisible();

    // 7. Mock Report endpoint
    await page.route('**/api/report/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/pdf',
        headers: {
          'Content-Disposition': 'attachment; filename="banking_report_dummy_data.pdf"',
        },
        body: Buffer.from('%PDF-1.4 Mock PDF Content %%EOF'),
      });
    });

    // 8. Navigate to reports and trigger download
    await page.goto('/dashboard/reports');
    await expect(page.locator('h2')).toContainText('Machine Learning & Analytics Reports');

    const downloadPromise = page.waitForEvent('download');
    await page.locator('button:has-text("Export AI PDF")').click();

    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/\.pdf$/);
  });
});
