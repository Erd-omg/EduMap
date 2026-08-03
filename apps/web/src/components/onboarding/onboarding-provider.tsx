'use client';

import { WelcomeOverlay } from './welcome-overlay';

/**
 * Client-side provider that renders the onboarding overlay.
 * This component is imported by the root layout (server component)
 * to add client-side functionality without converting the entire layout.
 */
export function OnboardingProvider() {
  return <WelcomeOverlay />;
}
