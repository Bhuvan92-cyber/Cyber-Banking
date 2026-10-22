import { test, expect } from '@playwright/test';

test.describe('Auth Routing, RBAC & Logout UX', () => {
  test('redirects unauthenticated user from /dashboard to /login without hitting landing page', async ({ page }) => {
    await page.goto('/dashboard');
    await expect(page).toHaveURL(/\/login\?(redirect=dashboard|from=%2Fdashboard)/);
    await expect(page.locator('h2')).toContainText('Sign in to your account');
  });

  test('redirects unauthenticated user from /admin to /login without hitting landing page', async ({ page }) => {
    await page.goto('/admin');
    await expect(page).toHaveURL(/\/login\?(redirect=admin|from=%2Fadmin)/);
    await expect(page.locator('h2')).toContainText('Sign in to your account');
  });

  test('logout from dashboard redirects to /login?redirect=dashboard, not public homepage', async ({ page }) => {
    // 1. Mock session / API
    await page.route('**/api/auth/me/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          isAuthenticated: true,
          user: { id: 1, username: 'analyst', is_staff: false, is_superuser: false },
        }),
      });
    });

    await page.route('**/api/auth/logout/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Logout successful.' }),
      });
    });

    await page.context().addCookies([
      {
        name: 'sessionid',
        value: 'active_session_token',
        domain: 'localhost',
        path: '/',
      },
    ]);

    await page.goto('/dashboard');
    await expect(page.locator('text=Sign Out')).toBeVisible();

    // Click Sign Out
    await page.click('text=Sign Out');

    // Verify redirected directly to /login?redirect=dashboard, never to /
    await expect(page).toHaveURL('/login?redirect=dashboard');
    await expect(page.locator('h2')).toContainText('Sign in to your account');
  });

  test('logout from admin portal redirects to /login?redirect=admin, not public homepage', async ({ page }) => {
    // Mock superadmin
    await page.route('**/api/auth/me/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          isAuthenticated: true,
          user: { id: 1, username: 'adminuser', is_staff: true, is_superuser: true },
        }),
      });
    });

    await page.route('**/api/auth/logout/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Logout successful.' }),
      });
    });

    await page.context().addCookies([
      {
        name: 'sessionid',
        value: 'active_admin_session_token',
        domain: 'localhost',
        path: '/',
      },
    ]);

    await page.goto('/admin');
    await expect(page.locator('text=Logout')).toBeVisible();

    // Click Logout
    await page.click('text=Logout');

    // Verify redirected directly to /login?redirect=admin
    await expect(page).toHaveURL('/login?redirect=admin');
    await expect(page.locator('h2')).toContainText('Sign in to your account');
  });

  test('RBAC: non-staff user attempting to access /admin is redirected to /dashboard', async ({ page }) => {
    await page.route('**/api/auth/me/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          isAuthenticated: true,
          user: { id: 2, username: 'standard_client', is_staff: false, is_superuser: false },
        }),
      });
    });

    await page.context().addCookies([
      {
        name: 'sessionid',
        value: 'standard_session',
        domain: 'localhost',
        path: '/',
      },
    ]);

    await page.goto('/admin');
    // Non-staff user should be redirected to /dashboard
    await expect(page).toHaveURL('/dashboard');
  });
});

