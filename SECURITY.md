# Security policy

## API keys

Never commit Mistral API keys, `.env` files, private keys, access tokens, or
credentials to this repository. The library reads `MISTRAL_API_KEY` from the
process environment. It deliberately does not load `.env` automatically.

If a key has appeared in a chat, issue, commit, terminal recording, or log,
consider it compromised: revoke it in the provider console and create a new
key. Removing the text later does not make the old key safe.

For GitHub Actions, store the value as a repository or environment secret named
`MISTRAL_API_KEY` and pass it only to the step that needs it:

```yaml
env:
  MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
```

Do not put a real value in `.env.example`; it exists only to document the
variable name.
