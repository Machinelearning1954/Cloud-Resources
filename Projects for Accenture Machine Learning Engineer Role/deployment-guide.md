# Deployment Guide

Comprehensive guide for deploying federal machine learning projects to production environments.

## Deployment Environments

### AWS GovCloud

AWS GovCloud provides FedRAMP High authorized cloud services for federal workloads.

#### Prerequisites

- AWS GovCloud account with appropriate permissions
- AWS CLI configured for GovCloud region
- Docker images pushed to Amazon ECR
- IAM roles and policies configured

#### ECS Deployment

```bash
# Login to ECR
aws ecr get-login-password --region us-gov-west-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-gov-west-1.amazonaws.com

# Build and tag image
docker build -t army-vehicle-maintenance:latest .
docker tag army-vehicle-maintenance:latest \
  <account-id>.dkr.ecr.us-gov-west-1.amazonaws.com/army-vehicle-maintenance:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-gov-west-1.amazonaws.com/army-vehicle-maintenance:latest

# Create ECS task definition
aws ecs register-task-definition \
  --cli-input-json file://ecs-task-definition.json \
  --region us-gov-west-1

# Create or update service
aws ecs create-service \
  --cluster federal-ml-cluster \
  --service-name army-vehicle-maintenance \
  --task-definition army-vehicle-maintenance:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=DISABLED}" \
  --region us-gov-west-1
```

#### Lambda Deployment

For serverless inference:

```bash
# Package Lambda function
pip install -r requirements.txt -t package/
cd package && zip -r ../lambda-function.zip . && cd ..
zip -g lambda-function.zip lambda_handler.py

# Create Lambda function
aws lambda create-function \
  --function-name vehicle-failure-predictor \
  --runtime python3.11 \
  --role arn:aws-us-gov:iam::<account-id>:role/lambda-execution-role \
  --handler lambda_handler.predict \
  --zip-file fileb://lambda-function.zip \
  --timeout 30 \
  --memory-size 1024 \
  --region us-gov-west-1

# Create API Gateway
aws apigatewayv2 create-api \
  --name vehicle-prediction-api \
  --protocol-type HTTP \
  --target arn:aws-us-gov:lambda:us-gov-west-1:<account-id>:function:vehicle-failure-predictor \
  --region us-gov-west-1
```

### Azure Government

Azure Government provides FedRAMP High and DoD IL5 authorized services.

#### Prerequisites

- Azure Government subscription
- Azure CLI configured for government cloud
- Container Registry in Azure Government
- Managed Identity or Service Principal

#### AKS Deployment

```bash
# Login to Azure Government
az cloud set --name AzureUSGovernment
az login

# Create resource group
az group create \
  --name federal-ml-rg \
  --location usgovvirginia

# Create AKS cluster
az aks create \
  --resource-group federal-ml-rg \
  --name federal-ml-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-managed-identity \
  --network-plugin azure \
  --network-policy azure \
  --generate-ssh-keys

# Get credentials
az aks get-credentials \
  --resource-group federal-ml-rg \
  --name federal-ml-cluster

# Deploy application
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/ingress.yml

# Verify deployment
kubectl get pods -n army-vehicle-maintenance
kubectl get services -n army-vehicle-maintenance
```

#### Azure Container Instances

For simpler deployments:

```bash
# Create container instance
az container create \
  --resource-group federal-ml-rg \
  --name army-vehicle-maintenance \
  --image <registry>.azurecr.us/army-vehicle-maintenance:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server <registry>.azurecr.us \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label army-vehicle-maintenance \
  --ports 8000 \
  --environment-variables \
    ENCRYPTION_ENABLED=true \
    AUDIT_LOG=true

# Get container IP
az container show \
  --resource-group federal-ml-rg \
  --name army-vehicle-maintenance \
  --query ipAddress.fqdn
```

## Kubernetes Deployment

### Namespace Configuration

```yaml
# k8s/namespace.yml
apiVersion: v1
kind: Namespace
metadata:
  name: army-vehicle-maintenance
  labels:
    name: army-vehicle-maintenance
    compliance: nist-800-171
```

### Deployment Configuration

```yaml
# k8s/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: predictive-maintenance
  namespace: army-vehicle-maintenance
spec:
  replicas: 3
  selector:
    matchLabels:
      app: predictive-maintenance
  template:
    metadata:
      labels:
        app: predictive-maintenance
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: api
        image: <registry>/army-vehicle-maintenance:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENCRYPTION_ENABLED
          value: "true"
        - name: AUDIT_LOG
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
```

### Service Configuration

```yaml
# k8s/service.yml
apiVersion: v1
kind: Service
metadata:
  name: predictive-maintenance-service
  namespace: army-vehicle-maintenance
spec:
  type: LoadBalancer
  selector:
    app: predictive-maintenance
  ports:
  - protocol: TCP
    port: 443
    targetPort: 8000
```

### Ingress Configuration

```yaml
# k8s/ingress.yml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: predictive-maintenance-ingress
  namespace: army-vehicle-maintenance
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.army-vehicle-maintenance.gov
    secretName: tls-secret
  rules:
  - host: api.army-vehicle-maintenance.gov
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: predictive-maintenance-service
            port:
              number: 443
```

## Security Configuration

### Secrets Management

#### AWS Secrets Manager

```bash
# Create secret
aws secretsmanager create-secret \
  --name army-vehicle-maintenance/api-key \
  --secret-string '{"api_key":"your-secret-key"}' \
  --region us-gov-west-1

# Retrieve secret in application
import boto3
import json

client = boto3.client('secretsmanager', region_name='us-gov-west-1')
response = client.get_secret_value(SecretId='army-vehicle-maintenance/api-key')
secret = json.loads(response['SecretString'])
api_key = secret['api_key']
```

#### Azure Key Vault

```bash
# Create key vault
az keyvault create \
  --name federal-ml-keyvault \
  --resource-group federal-ml-rg \
  --location usgovvirginia

# Add secret
az keyvault secret set \
  --vault-name federal-ml-keyvault \
  --name api-key \
  --value "your-secret-key"

# Retrieve secret in application
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

credential = DefaultAzureCredential()
client = SecretClient(
    vault_url="https://federal-ml-keyvault.vault.usgovcloudapi.net/",
    credential=credential
)
secret = client.get_secret("api-key")
api_key = secret.value
```

### TLS/SSL Configuration

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=api.army-vehicle-maintenance.gov"

# Create Kubernetes secret
kubectl create secret tls tls-secret \
  --cert=tls.crt \
  --key=tls.key \
  -n army-vehicle-maintenance

# For production, use cert-manager with Let's Encrypt
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus-config.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'army-vehicle-maintenance'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - army-vehicle-maintenance
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: predictive-maintenance
```

### Grafana Dashboards

Import pre-built dashboards for:
- Model performance metrics
- API latency and throughput
- Error rates and status codes
- Resource utilization (CPU, memory)
- Data drift detection

### Log Aggregation

#### ELK Stack

```bash
# Deploy Elasticsearch
kubectl apply -f https://download.elastic.co/downloads/eck/2.9.0/crds.yaml
kubectl apply -f https://download.elastic.co/downloads/eck/2.9.0/operator.yaml

# Deploy Kibana
kubectl apply -f k8s/elasticsearch.yml
kubectl apply -f k8s/kibana.yml

# Configure Filebeat for log collection
kubectl apply -f k8s/filebeat.yml
```

## Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: predictive-maintenance-hpa
  namespace: army-vehicle-maintenance
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: predictive-maintenance
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Pod Autoscaler

```yaml
# k8s/vpa.yml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: predictive-maintenance-vpa
  namespace: army-vehicle-maintenance
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: predictive-maintenance
  updatePolicy:
    updateMode: "Auto"
```

## Backup and Disaster Recovery

### Model Backup

```bash
# Backup models to S3
aws s3 sync models/ s3://federal-ml-backups/models/ \
  --region us-gov-west-1 \
  --sse AES256

# Backup to Azure Blob Storage
az storage blob upload-batch \
  --destination models \
  --source models/ \
  --account-name federalmlstorage
```

### Database Backup

```bash
# Automated RDS snapshots (AWS)
aws rds create-db-snapshot \
  --db-instance-identifier federal-ml-db \
  --db-snapshot-identifier federal-ml-db-snapshot-$(date +%Y%m%d) \
  --region us-gov-west-1

# Azure SQL Database backup
az sql db export \
  --resource-group federal-ml-rg \
  --server federal-ml-server \
  --name federal-ml-db \
  --admin-user admin \
  --admin-password <password> \
  --storage-key <storage-key> \
  --storage-key-type StorageAccessKey \
  --storage-uri https://federalmlstorage.blob.core.usgovcloudapi.net/backups/db-backup.bacpac
```

## Rollback Procedures

### Kubernetes Rollback

```bash
# View deployment history
kubectl rollout history deployment/predictive-maintenance -n army-vehicle-maintenance

# Rollback to previous version
kubectl rollout undo deployment/predictive-maintenance -n army-vehicle-maintenance

# Rollback to specific revision
kubectl rollout undo deployment/predictive-maintenance \
  --to-revision=2 \
  -n army-vehicle-maintenance
```

### ECS Rollback

```bash
# Update service to previous task definition
aws ecs update-service \
  --cluster federal-ml-cluster \
  --service army-vehicle-maintenance \
  --task-definition army-vehicle-maintenance:1 \
  --region us-gov-west-1
```

## Compliance Checklist

Before production deployment:

- [ ] Security scan completed with no critical vulnerabilities
- [ ] Penetration testing performed and issues remediated
- [ ] NIST 800-171 controls implemented and documented
- [ ] Encryption enabled for data at rest and in transit
- [ ] Audit logging configured and tested
- [ ] Access controls (RBAC, MFA) configured
- [ ] Secrets managed through secure vault (not hardcoded)
- [ ] Network policies configured (least privilege)
- [ ] Monitoring and alerting configured
- [ ] Backup and disaster recovery tested
- [ ] Incident response procedures documented
- [ ] Compliance documentation reviewed and approved
- [ ] Authority to Operate (ATO) obtained (if required)

## Troubleshooting

### Common Issues

**Issue**: Container fails to start
```bash
# Check logs
kubectl logs -f deployment/predictive-maintenance -n army-vehicle-maintenance

# Check events
kubectl get events -n army-vehicle-maintenance --sort-by='.lastTimestamp'
```

**Issue**: High latency
```bash
# Check resource utilization
kubectl top pods -n army-vehicle-maintenance

# Scale up replicas
kubectl scale deployment/predictive-maintenance --replicas=5 -n army-vehicle-maintenance
```

**Issue**: Model loading errors
```bash
# Verify model files exist
kubectl exec -it <pod-name> -n army-vehicle-maintenance -- ls -la /app/models/

# Check model file integrity
kubectl exec -it <pod-name> -n army-vehicle-maintenance -- \
  python -c "import joblib; joblib.load('/app/models/xgboost_model.pkl')"
```

## Support

For deployment assistance:
- Review project documentation
- Check GitHub issues for similar problems
- Contact DevOps team for infrastructure support
- Escalate to security team for compliance questions

---

**Last Updated**: December 2025  
**Version**: 1.0
