Deploying this Streamlit app to Fly.io

Prerequisites
- Install flyctl: https://fly.io/docs/hands-on/install-flyctl/
- Sign up / login: `flyctl auth login`
- Have a docker-capable machine (or let Fly build using buildpacks)

Quick deploy (recommended)
1. From your project root, login and create an app (choose region when prompted):

   flyctl launch --name dsfmlproject --region ord

   - If prompted about VM vs Dockerfile, choose Dockerfile (we added one).
   - This will create a local `fly.toml` (you can keep or replace it).

2. Build and deploy:

   flyctl deploy

3. Open the app (fly prints URL) or run:

   flyctl open

Notes
- The Dockerfile copies the local `artifacts/` directory into the image at build time. If your CSV/model artifacts change, rebuild & redeploy or move artifacts to S3 and update the app to read from S3.
- For persistent storage or frequent artifact updates, configure an object store (S3/GCS) and set credentials via `flyctl secrets set`.

Setting secrets (example for AWS S3):

   flyctl secrets set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy

Scaling & resource tips
- If your model is large or uses more RAM, increase the VM size:

   flyctl scale memory 1GiB

- For production traffic, add more instances or set autoscaling.

Rollback
- Fly keeps previous releases; use `flyctl releases` and `flyctl rollback <id>` to revert.
