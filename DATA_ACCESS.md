# Data Access and Licensing

## Important restriction

This repository does not redistribute TwiBot-20, TwiBot-22, Twitter/X content, user-level predictions, profile text, graph edges, model embeddings, or checkpoints derived from restricted source data.

The MIT license covers only the original software and documentation. Dataset files remain governed by their original authors' access requirements and applicable platform terms.

## TwiBot access

The TwiBot authors require users to join the official access group:

- Access group: https://groups.google.com/g/twibot_access
- TwiBot-20 paper: https://arxiv.org/abs/2106.13088
- TwiBot-22 paper: https://arxiv.org/abs/2206.04564
- TwiBot-20 data: https://drive.google.com/drive/folders/1YY_fFK-ZS0lFDvoLaijoCp0yQ1CnG33i
- TwiBot-22 data: https://drive.google.com/drive/folders/1YwiOUwtl8pCd2GD97Q_WEzwEUtSPoxFs

Access permission is controlled by the dataset owners, not this repository.

## TwiBot-22 files used

The source experiment requires `user.json`, `label.csv`, `split.csv`, and `edge.csv`.

Expected sizes and SHA-256 hashes are recorded in `manifests/input_manifest.json`. Verify them with:

    Get-FileHash ..\user.json,..\label.csv,..\split.csv,..\edge.csv -Algorithm SHA256

The hashes must match the manifest. A mismatch indicates a different release, incomplete download, or modification.

## TwiBot-20 Format-22 files used

The external evaluation expects:

    TwiBot-20-Format22\Twibot-20\node.json
    TwiBot-20-Format22\Twibot-20\label.csv
    TwiBot-20-Format22\Twibot-20\split.csv
    TwiBot-20-Format22\Twibot-20\edge.csv

Script 29 computes provenance hashes and audits labeled users, official splits, field availability, and graph availability before evaluation.

## Local configuration

Machine-specific paths belong only in the ignored file `configs/config.local.yaml`.

Never commit local paths, credentials, access tokens, private URLs, or downloaded datasets.

## Excluded content

Raw datasets, generated outputs, logs, checkpoints, model weights, embeddings, NumPy arrays, Parquet files, serialized preprocessors, and user-level predictions are excluded from version control.

Only compact aggregate evidence intended for manuscript verification is included in `evidence/`.

## User responsibilities

Users must obtain data through authorized channels, follow original citation and usage requirements, comply with Twitter/X terms and applicable research-ethics requirements, avoid republishing identifiable account-level content, and independently verify that their intended use is permitted.
