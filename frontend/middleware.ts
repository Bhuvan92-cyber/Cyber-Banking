import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';


/**
 * Next.js Edge Middleware for route protection
 * Checks for Django's 'sessionid' cookie.
 * If missing when requesting /dashboard or /admin, redirects to /login.
 */
export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const sessionid = request.cookies.get('sessionid')?.value;

  const isProtected = pathname.startsWith('/dashboard') || pathname.startsWith('/admin');

  if (isProtected && !sessionid) {
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('from', pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/dashboard/:path*', '/admin/:path*'],
};
