# Disaster Recovery & Resilience Runbook
## AI-Powered Urban Air Quality Intelligence Platform

This engineering runbook specifies the exact programmatic recovery mechanisms and standard operating procedures for resolving infrastructure failure states in the production cluster.

### 1. API Outages: CAAQMS/Satellite Ingestion Failure
**Failure State:** The external CAAQMS API or Satellite Data providers return HTTP 429 (Rate Limit) or complete 5xx server failures.
**Recovery Mechanism (Redis Fallback Caches):**
- The FastAPI ingestion workers utilize an automated `asyncio` exponential backoff (2s -> 64s).
- If the backoff threshold is exceeded, the circuit breaker opens.
- The system automatically fails over to the **Localized Redis Fallback Cache**. The Redis cluster retains the T-1 hour spatial grid states. 
- The Predictive Forecasting Agent is designed to seamlessly ingest these cached state vectors, calculating forward predictions with slightly wider confidence intervals without blocking the main event loop.
- **SRE Action:** No manual action required. Check `Grafana` for circuit breaker status.

### 2. Database Pressure: TimescaleDB Cluster Drops
**Failure State:** Sudden micro-bursts of IoT data overwhelm the TimescaleDB insertion connection pools, leading to transaction timeouts.
**Recovery Mechanism (Kafka Topic Message Retention):**
- TimescaleDB writes are decoupled via Apache Kafka. 
- If TimescaleDB goes down, the Kafka broker configuration `log.retention.hours=168` ensures up to 7 days of raw spatio-temporal data is retained on disk.
- Consumer groups will automatically pause offset commits.
- Once the TimescaleDB cluster is restarted or auto-scaled horizontally by the Kubernetes operator, the Kafka consumers resume and batch-insert the backlog using bulk `COPY` operations.
- **SRE Action:** If TimescaleDB pods are in `CrashLoopBackOff`, evaluate PVC sizing and bump max connections in `postgresql.conf`.

### 3. Memory Leaks & Failures: Celery Worker Crashes
**Failure State:** Heavy Spatial SHAP (Causal Attribution) matrix calculations cause Celery worker pods to exceed their Memory Limits (OOMKilled).
**Recovery Mechanism (Kubernetes Auto-Recovery):**
- The task state is set to `late_ack=True`. When a Celery worker is OOMKilled by the Kubelet, the message is NOT acknowledged and is returned to the RabbitMQ/Redis broker.
- The Kubernetes Horizontal Pod Autoscaler (HPA) detects memory pressure and provisions new Celery worker pods on secondary nodes.
- The failed SHAP calculation is retried by a newly provisioned worker with a pristine memory space.
- **SRE Action:** If specific grids consistently trigger OOM, tune the Optuna hyperparameter bounds in `core_tasks.py` to reduce GNN tree depth, or allocate dedicated GPU node taints.
