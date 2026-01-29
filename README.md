# smolpuff 🐿️

S3 based search engine designed (inspired by [Turbopuffer](https://turbopuffer.com/)) for fun!

## Using with LocalStack

Run LocalStack to emulate S3 locally and point smolpuff at it.

1. Start LocalStack (Docker):

```sh
docker run --rm -d --name localstack -p 4566:4566 -e SERVICES=s3 localstack/localstack
# Or pip:
pip install localstack
localstack start
```

2. Install awslocal (optional, convenient helper):

```sh
pip install awscli-local
```

3. Create the S3 bucket:

```sh
awslocal s3 mb s3://smolpuff --region us-east-1
# Or with AWS CLI:
aws --endpoint-url http://localhost:4566 s3 mb s3://smolpuff --region us-east-1
```

4. (Optional) Export AWS env vars if you prefer the app to read them:

```sh
export AWS_ACCESS_KEY_ID=access_key_id
export AWS_SECRET_ACCESS_KEY=secret_access_key
export AWS_REGION=us-east-1
```

The code in `src/main.rs` currently uses `http://localhost:4566` and example credentials
(see the S3 builder block) — update if you changed ports/credentials.

5. Run the app:

```sh
cargo run
```

You should see logs about adding vectors and query results. If you need to create or inspect buckets/objects, use `awslocal` or `aws --endpoint-url`.
