import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/agent/:path*',
        destination: `${process.env.AGENT_API_URL || 'http://localhost:8000'}/:path*`,
      },
    ]
  },
}

export default nextConfig
