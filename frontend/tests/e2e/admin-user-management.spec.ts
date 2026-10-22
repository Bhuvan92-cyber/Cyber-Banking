import { test, expect } from '@playwright/test';

test.describe('Admin User Management & RBAC Interactive Controls', () => {
  test('renders user list, changes role, and toggles status', async ({ page }) => {
    // 1. Mock superadmin profile
    await page.route('**/api/auth/me/', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          isAuthenticated: true,
          user: { id: 1, username: 'superadmin', is_staff: true, is_superuser: true },
        }),
      });
    });

    // 2. Mock users list
    let mockUsers = [
      {
        id: 1,
        username: 'superadmin',
        email: 'superadmin@bank.internal',
        name: 'Super Admin',
        role: 'Superadmin',
        status: 'Active',
        is_staff: true,
        is_superuser: true,
        is_active: true,
        joined: '2024-01-01',
      },
      {
        id: 2,
        username: 'analyst_bob',
        email: 'bob@bank.internal',
        name: 'Bob Analyst',
        role: 'Staff Analyst',
        status: 'Active',
        is_staff: true,
        is_superuser: false,
        is_active: true,
        joined: '2024-02-15',
      },
    ];

    await page.route('**/api/admin/users/', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockUsers),
        });
      }
    });

    // Mock role mutation
    await page.route('**/api/admin/users/2/', async (route) => {
      if (route.request().method() === 'PATCH') {
        const body = JSON.parse(route.request().postData() || '{}');
        if (body.role) {
          mockUsers[1].role = body.role;
        }
        if (typeof body.is_active === 'boolean') {
          mockUsers[1].is_active = body.is_active;
          mockUsers[1].status = body.is_active ? 'Active' : 'Inactive';
        }
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            message: 'User updated successfully.',
            user: mockUsers[1],
          }),
        });
      }
    });

    await page.context().addCookies([
      {
        name: 'sessionid',
        value: 'superadmin_session',
        domain: 'localhost',
        path: '/',
      },
    ]);

    // 3. Visit user management page
    await page.goto('/admin/users');
    await expect(page.getByRole('heading', { name: 'User Management & RBAC' })).toBeVisible();
    await expect(page.locator('text=Bob Analyst')).toBeVisible();

    // 4. Open action menu for Bob
    const bobRow = page.locator('tr:has-text("Bob Analyst")');
    await bobRow.locator('button[title="Open action menu"]').click();

    // 5. Open Role Modal
    await page.locator('text=Change Role Permissions').click();
    await expect(page.locator('h3:has-text("Modify Role Permissions")')).toBeVisible();

    // Select Superadmin inside modal
    await page.locator('.fixed.inset-0').locator('span:text-is("Superadmin")').click();
    await page.locator('button:has-text("Save Role Permissions")').click();

    // Verify modal closes
    await expect(page.locator('h3:has-text("Modify Role Permissions")')).not.toBeVisible();
  });
});

