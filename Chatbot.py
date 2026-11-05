import numpy as np
import json
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam

def process_input(user_text, tokenizer, max_len, model, Data, classes, threshold=0.5):
  seq = tokenizer.texts_to_sequences([user_text])
  seq_padded = pad_sequences(seq, maxlen=max_len-1, padding='pre')
  predictions = model.predict(seq_padded, verbose=0)[0]
  max_prob = np.max(predictions)
  predicted_index = np.argmax(predictions)

  if max_prob < threshold:
    return "I'm not sure how to respond to that."
    
  predicted_intent = classes[predicted_index]
  for intent in Data['intents']:
    if intent['intent'] == predicted_intent:
      return intent['response']
  return "Sorry, I didn't understand that."

if __name__ == "__main__":
  with open('data.json', 'r') as f:
    Data = json.load(f)

  query = []
  documents = []
  classes = []

  for intent in Data["intents"]:
    for example in intent["examples"]:
      query.append(example)
      if intent["intent"] not in classes:
        classes.append(intent["intent"])
      documents.append((example, intent["intent"]))

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(query)

  total_words = len(tokenizer.word_index) + 1
  max_len = 100

  with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
  with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

  query = tokenizer.texts_to_sequences(query)
  query = np.array(pad_sequences(query, maxlen=max_len))

  output = np.zeros((len(documents), len(classes)))
  for i, (_, intent) in enumerate(documents):
    output[i][classes.index(intent)] = 1

  model = Sequential()
  model.add(Embedding(total_words, 200, input_length=max_len-1))
  model.add(Bidirectional(LSTM(200)))
  model.add(Dropout(0.5))
  model.add(Dense(200, activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(len(classes), activation='softmax'))
  adam = Adam(learning_rate=0.005)
  model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

  hist = model.fit(query, np.array(output), epochs=200, batch_size=750, verbose=2)

  model.save("Chatbot.keras")

  print("Type 'exit' to Exit\nChatbot: Hello! How can I assist you today?")
  while True:
    user_text = input("You : ")
    if user_text.lower() == "exit":
      break
    print("Chatbot:", process_input(user_text, tokenizer, max_len, model, Data, classes))