# Next.js 15 standalone Dockerfile
FROM node:22-alpine AS base

# Install pnpm
RUN corepack enable && corepack prepare pnpm@9 --activate

# Build stage
FROM base AS deps
WORKDIR /app
COPY pnpm-workspace.yaml package.json pnpm-lock.yaml ./
COPY apps/web/package.json ./apps/web/package.json
COPY packages/shared-types/package.json ./packages/shared-types/package.json
RUN pnpm install --frozen-lockfile

# Development stage
FROM deps AS dev
WORKDIR /app
COPY . .
EXPOSE 3000
CMD ["pnpm", "dev"]

# Production build stage
FROM deps AS builder
WORKDIR /app
COPY . .
RUN pnpm build

# Production stage
FROM node:22-alpine AS prod
WORKDIR /app
COPY --from=builder /app/apps/web/.next ./apps/web/.next
COPY --from=builder /app/apps/web/public ./apps/web/public
COPY --from=builder /app/apps/web/package.json ./apps/web/
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["pnpm", "--filter", "@edumap/web", "start"]
