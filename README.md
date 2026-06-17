#AURORA - Auxiliary Uniform Response Observation Research Algorithm

This is a simple GRU model text generator, it can read text and use it to train the model, and can then be used to generate new responses.

Text is read from training_data.txt, and converts it into tokens with subword.model, and subword.vocab and is stored in a compressed file dataset.npz.
upon training and testing, the parameters are saved to and loaded fromj params.json.
The training data currently consists of public domain books.

This project features a text tokenizer, embedding, GRU RNN, and probability token selection.

You can run the program and enter either "test" or "train" to use the model.

recomeded version: Python 3.13 (tested)
required libraries:
  tensorflow,
  numpy,
  json,
  sentencepiece
