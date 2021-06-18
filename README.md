# 日本語RoBERTa-base
日本語Wikipediaで学習したRoBERTa-baseのサンプルコードです。

## 学習設定

fairseqを用いて学習したものをpytorch用に変換しました。
学習に使用したスクリプトは以下の通りです。

~~~bash:train.sh
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.00005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=32        # Number of sequences per batch (batch size)
UPDATE_FREQ=2        # Increase the batch size

DATA_DIR=data/wiki201221_janome_vocab_32000/data-bin

fairseq-train --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --save-dir model/roberta_base_wiki201221_janome_vocab_32000 --save-interval 10 \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 \
    --skip-invalid-size-inputs-valid-test
~~~

## 使用方法

必要なモデルを[Google Drive](https://drive.google.com/file/d/1cMIED6Yt38WSBXyhTpE8rlMQjLuWzGol/view?usp=sharing)からダウンロードし、`model`下に置いてください。

~~~bash
pip install -r requirements.txt
python3 main.py
~~~
