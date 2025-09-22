# Japanese Netflix 1K deck builder

This repository contains data and helper scripts used to build a 1k Japanese Netflix frequency list. The list originally comes from https://ankiweb.net/shared/info/93196238 but was updated to include example sentences and images.

It's built on the n+1 principle, meaning that each row is built on the previous rows and tries to only add one new piece of vocabulary.

To achieve this, for each row, we send our entire current vocabulary to GPT-5 and ask it to generate a sentence that includes the new word. We then use that sentence to generate an image. Later on, in Anki, you can use HyperTTS to generate the audio for the word and sentences.

AI is not perfect (yet, at least), and so mistakes can slip in. Additionally, images generated can contain mistakes or be inappropriate. So you'll have to manually review the deck.

**Finally be careful with the cost**: Because we send the full current vocabulary for each row and because thinking is enabled, the cost is quite high. For GPT-5, expect around $25 to build the deck. You can reduce the cost by limiting thinking or switching to a cheaper model.

## Usage

You can download the deck from [https://ankiweb.net/shared/info/1925323165](https://ankiweb.net/shared/info/1925323165).

## Development

Add to your `.env` file:

```
NOTES_PATH=notes.txt
KNOWN_PATH=known-word.txt
OUTPUT_PATH=flashcards.csv
START_INDEX=1
LIMIT=0
OPENAI_MODEL=gpt-5
OPENAI_REASONING_EFFORT=medium

OPENAI_ACCESS_TOKEN=ADD YOUR TOKEN HERE
REPLICATE_API_TOKEN=ADD YOUR TOKEN HERE
```

```bash
bundle install
bundle exec ruby text.rb
bundle exec ruby images.rb
```

Once finished, you will need to import the CSV into Anki and move the images to the media folder.
