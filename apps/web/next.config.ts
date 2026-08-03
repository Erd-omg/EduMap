import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  transpilePackages: ['@edumap/shared-types'],
  output: 'standalone',
};

export default nextConfig;
