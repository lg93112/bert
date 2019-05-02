## BERT for mood detection task

1. Download pre-trained model from google, and put to a folder(e.g.BERT_BASE_DIR). We use BERT-Base, Uncased pretrained model.

2. Secondly all data have been put under Mood directory

3. To fine-tune and train the BERT model, run following command:
python run_classifier.py \
  --task_name=mood \
  --do_train=true \
  --do_eval=true \
  --data_dir=$MOOD \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32\
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --output_dir=outputs/mood/
  
4. To test BERT model on test examples, run following command:
 python run_classifier.py \
  --task_name=mood \
  --do_predict=true \
  --data_dir=$MOOD \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=outputs/mood/
