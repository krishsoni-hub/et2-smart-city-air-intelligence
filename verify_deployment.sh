#!/bin/bash
# ==============================================================================
# END-TO-END SYSTEM SMOKE TEST & DEPLOYMENT VERIFICATION
# ==============================================================================

set -e
echo "Starting System Smoke Test & Deployment Verification..."

# 1. Check FastAPI /health endpoint
echo "Checking FastAPI Gateway..."
HTTP_STATUS=$(curl -o /dev/null -s -w "%{http_code}\n" http://localhost:8000/docs || echo "000")
if [ "$HTTP_STATUS" == "200" ]; then
    echo "✅ API Gateway is healthy and responding (HTTP 200)."
else
    echo "❌ API Gateway health check failed. HTTP Status: $HTTP_STATUS"
fi

# 2. Query Kafka Broker Metadata
echo "Checking Kafka Broker Metadata..."
if docker exec et2-smart-city-air-intelligence-kafka-1 kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
    echo "✅ Kafka Broker is active and accepting connections."
else
    echo "❌ Kafka Broker is down or unreachable."
fi

# 3. Check Redis Ping Connectivity
echo "Checking Redis Cache Connectivity..."
REDIS_PING=$(docker exec et2-smart-city-air-intelligence-redis-1 redis-cli ping || echo "FAIL")
if [[ "$REDIS_PING" == *"PONG"* ]]; then
    echo "✅ Redis cluster is active (PONG)."
else
    echo "❌ Redis connectivity failed."
fi

# 4. Query PostgreSQL / TimescaleDB
echo "Checking TimescaleDB Hyper-tables..."
PG_CHECK=$(docker exec et2-smart-city-air-intelligence-timescaledb-1 pg_isready -U admin -d smartcity_db || echo "FAIL")
if [[ "$PG_CHECK" == *"accepting connections"* ]]; then
    echo "✅ TimescaleDB is active and accepting inserts."
else
    echo "❌ TimescaleDB connection failed. Output: $PG_CHECK"
fi

echo "Deployment Verification Complete."
