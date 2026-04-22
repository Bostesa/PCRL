# BLOCKED: AWS session expired — instance still running

**Action needed:** Stop AWS instance i-0cd861b61dc3ac842.

```bash
aws login
aws ec2 stop-instances --instance-ids i-0cd861b61dc3ac842
```

All sweep and final evaluation jobs are complete. The instance is idle and costing money.
