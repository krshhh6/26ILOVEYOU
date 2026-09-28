# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package manifests and install dependencies
COPY package.json package-lock.json* ./
RUN npm ci --legacy-peer-deps

# Copy application source and build production bundle
COPY . .
RUN npm run build

# Production serve stage with Nginx
FROM nginx:1.25-alpine

# Copy built static assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration with API reverse proxy
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
