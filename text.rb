#!/usr/bin/env ruby

# Build flashcards.csv by generating example sentences with OpenAI (GPT‑5)
# - Reads words from notes.txt (tab-separated: JP, EN, JP_Furigana)
# - Uses known words from known-word.txt plus all previous rows (n+1 principle)
# - Calls OpenAI Responses API (ruby-openai gem) with medium reasoning
# - Writes/updates flashcards.csv with columns:
#   ID, JP, EN, JP_Furigana, Pronounciation, Sentence JP, Sentence EN, Sentence Romaji, Sentence pronounciation, Explanation

# -----------------------
# Config
# -----------------------
require "dotenv"
require "openai"
require "csv"
require "json"

Dotenv.load

MODEL = ENV.fetch("OPENAI_MODEL", "gpt-5")
OPENAI_URI_BASE = ENV.fetch("OPENAI_URI_BASE", "https://api.openai.com/v1")
REASONING_EFFORT = ENV.fetch("OPENAI_REASONING_EFFORT", "medium")
INPUT_NOTES = ENV.fetch("NOTES_PATH")
INPUT_KNOWN = ENV.fetch("KNOWN_PATH")
OUTPUT_CSV = ENV.fetch("OUTPUT_PATH")
START_INDEX = (ENV["START_INDEX"] || "1").to_i # 1-based row index in notes.txt
LIMIT = ENV["LIMIT"]&.to_i # optional: process only first N rows from START_INDEX

# Allow a small whitelist of grammar/function words even if not in known list (particles etc.)
ALLOWED_GRAMMAR = %w[
  は が を に へ で と も の か ね よ から まで だけ など だけど でも そして しかし だから ので のでした
  だ です でした じゃない じゃありません ます ました ません ましょう たい た て ている いる ある ですか ください
  この その あの ここ そこ あそこ これ それ あれ もう まだ とても すごく たくさん 少し ちょっと いつ どこ だれ 何 なん どう
  一 二 三 四 五 六 七 八 九 十 百 千 万 円 時 分 日 月 年 週 回 人 目
  。 、 ！ ？
].freeze

# -----------------------
# Helpers
# -----------------------

def configure_openai!
  OpenAI.configure do |config|
    # Prefer env var; do not log errors in production
    config.access_token = ENV.fetch("OPENAI_ACCESS_TOKEN")
    config.uri_base = OPENAI_URI_BASE
    config.log_errors = true
    config.uri_base = ENV["OPENAI_URI_BASE"] if ENV["OPENAI_URI_BASE"]
  end
end

def client
  @client ||= OpenAI::Client.new(request_timeout: 60 * 10)
end

def load_notes(path)
  lines = File.readlines(path, chomp: true)
  lines.map.with_index(1) do |line, idx|
    next if line.strip.empty?
    parts = line.split("\t")
    if parts.size < 2
      warn "Skipping malformed line #{idx} in #{path}: #{line.inspect}"
      next
    end
    jp = parts[0]&.strip || ""
    en = parts[1]&.strip || ""
    fur = parts[2]&.strip || jp
    {jp: jp, en: en, furigana: fur}
  end.compact
end

def load_known_words(path)
  return [] unless File.exist?(path)
  File.readlines(path, chomp: true).map(&:strip).reject(&:empty?)
end

def extract_pronunciation(text)
  return "" if text.nil?
  return text.dup unless text.include?("[") || text.include?("]")

  result = +""
  i = 0
  length = text.length

  while i < length
    char = text[i]

    if char == "["
      closing = text.index("]", i + 1)
      if closing
        result << text[(i + 1)...closing]
        i = closing
      end
    elsif char == "]"
      # skip stray closers
    elsif char.match?(/\p{Han}/)
      # skip kanji; their readings are collected inside brackets
    else
      result << char
    end

    i += 1
  end

  result
end

def strip_furigana(text)
  return "" if text.nil?

  cleaned = text.gsub(/\[[^\]]*\]/, "")
  cleaned.gsub(/\s+(?=[\p{Han}\p{Hiragana}\p{Katakana}])/u, "").strip
end

def build_prompt(row, known_list)
  known_preview = known_list.uniq
  allow = ALLOWED_GRAMMAR

  <<~PROMPT
    You are a Japanese tutor creating n+1 natural sounding example sentences for Anki.

    Task:
    - Sentence should be natural and sound like a real conversation.
    - Sentence should be as short as possible.
    - Use ONLY words from the allowed set below, plus the single target word.
    - Keep the sentence natural, simple (A1–A2), and short (4–8 tokens).
    - Must include the target JP exactly as given.
    - Avoid names, slang, rare kanji, or extra vocabulary outside the allowed set.
    - Output strictly JSON with the schema below and NOTHING else.
    - Include an optional "explanation" only when helpful for beginners (A1–A2): 1–2 short English sentences about meaning, very basic nuance, common use, politeness/casual level, or simple grammar note for this word. Do not just say that the word means x, as it is obvious. If not needed, just set it to an empty string.

    Target word:
    - JP: #{row[:jp]}
    - EN gloss: #{row[:en]}
    - JP with furigana: #{row[:furigana]}

    Allowed words (known):
    #{known_preview.join(", ")}

    Allowed grammar tokens (always OK even if not in known):
    #{allow.join(" ")}

    Output JSON schema:
    {
      "sentence_jp": "One Japanese sentence with Anki furigana 漢字[よみ] (例: 映画[えいが]). Only rule: if kana or punctuation immediately precedes a 漢字[よみ] group, insert one ASCII space before it (例: 世[よ]の 中[なか]、). Otherwise add no spaces (東京[とうきょう]駅[えき], 会[あ]う). Punctuation stays outside brackets. Output only the sentence.",
      "sentence_en": "Natural English translation.",
      "sentence_romaji": "Full romaji of the Japanese sentence (Hepburn, no diacritics).",
      "explanation": "Optional A1–A2 English explanation (1–2 sentences). Only include if helpful."
    }

    Constraints for sentence_jp:
    - Include the target word exactly as #{row[:jp]}.
    - For every kanji sequence, append its reading in hiragana inside [] immediately after the kanji, e.g., 新聞[しんぶん] を 読[よ]む.
    - Do NOT add furigana to kana-only words.
    - Use standard Japanese punctuation; avoid quotes.
  PROMPT
end

def call_openai_generate(row, known_words)
  prompt = build_prompt(row, known_words)

  response = client.responses.create(parameters: {
    model: MODEL,
    input: prompt,
    reasoning: {effort: REASONING_EFFORT}
  })

  # Surface API errors explicitly if present
  if response.is_a?(Hash) && response["error"]
    err = response["error"]
    message = err.is_a?(Hash) ? (err["message"] || err.inspect) : err.to_s
    raise "OpenAI API error: #{message}"
  end

  # The Responses API returns an array in "output"; collect text segments
  text_chunks = []
  if response.is_a?(Hash) && response["output"].is_a?(Array)
    response["output"].each do |item|
      next unless item.is_a?(Hash)
      content = item["content"]
      next unless content.is_a?(Array)
      content.each do |c|
        if c.is_a?(Hash) && (c["type"] == "output_text" || c["type"].nil?)
          text_chunks << c["text"].to_s
        end
      end
    end
  end

  raw = text_chunks.join("\n").strip
  parsed = parse_json_payload(raw)

  unless parsed
    # Try to salvage by extracting JSON substring
    if (start = raw.index("{")) && (finish = raw.rindex("}"))
      candidate = raw[start..finish]
      parsed = parse_json_payload(candidate)
    end
  end

  unless parsed
    raise "OpenAI response could not be parsed as JSON. Raw: #{raw[0, 500]}..."
  end

  # Basic validation
  sjp = parsed[:sentence_jp].to_s.strip
  unless sjp.include?(row[:jp])
    warn "Validation: sentence does not contain target '#{row[:jp]}'. Will still record but please verify."
  end

  parsed
end

def parse_json_payload(text)
  json = JSON.parse(text)
  # Accept either an object or a single-element array of objects
  json = json.first if json.is_a?(Array)
  return nil unless json.is_a?(Hash)

  {
    sentence_jp: json["sentence_jp"],
    sentence_en: json["sentence_en"],
    sentence_romaji: json["sentence_romaji"],
    explanation: json["explanation"]
  }
rescue JSON::ParserError
  nil
end

def open_csv_for_resume(path)
  exists = File.exist?(path)
  # Treat files that only contain whitespace/newlines as empty
  was_empty = if exists
    File.zero?(path) || File.read(path).strip.empty?
  else
    true
  end
  mode = exists ? "a" : "w"
  csv = CSV.open(path, mode, force_quotes: true)
  if !exists || was_empty
    # New or empty file: write header with ID as first column
    csv << ["ID", "Ranking", "JP", "EN", "JP Furigana", "Pronounciation", "Sentence JP", "Sentence EN", "Sentence Romaji", "Sentence pronounciation", "Explanation"]
    csv.flush
  end
  csv
end

def count_existing_rows(path)
  return 0 unless File.exist?(path)
  n = 0
  CSV.foreach(path) { |_| n += 1 }
  [n - 1, 0].max # exclude header
end

# -----------------------
# Main
# -----------------------

def main
  configure_openai!

  notes = load_notes(INPUT_NOTES)
  if notes.empty?
    warn "No notes found in #{INPUT_NOTES}"
    exit 1
  end

  known_seed = load_known_words(INPUT_KNOWN)
  puts "Loaded #{known_seed.size} seed known words from #{INPUT_KNOWN}"

  processed_existing = count_existing_rows(OUTPUT_CSV)
  start_idx = [START_INDEX, 1].max
  start_idx = processed_existing + 1 if processed_existing > 0 && START_INDEX <= 1
  total = notes.size

  last_idx = if LIMIT && LIMIT > 0
    [start_idx + LIMIT - 1, total].min
  else
    total
  end

  csv = open_csv_for_resume(OUTPUT_CSV)

  # Accumulating known words as we move forward
  cumulative_known = known_seed.dup

  # Next sequential ID continues across runs
  next_id = processed_existing + 1

  puts "Processing rows #{start_idx}..#{last_idx} of #{total}"

  (1..total).each do |i|
    row = notes[i - 1]
    # Add previous row JP entries to known list BEFORE generating for current i
    if i < start_idx
      cumulative_known << row[:jp]
      cumulative_known << row[:furigana]
      next
    end
    break if i > last_idx

    # Build known words set for this row: seed + all previous JP and furigana
    known_for_row = (cumulative_known + ALLOWED_GRAMMAR).uniq

    print "[#{i}/#{total}] #{row[:jp]} #{row[:en]}\n"
    payload = nil
    begin
      payload = call_openai_generate(row, known_for_row)
    rescue => e
      warn "Error for row #{i} (#{row[:jp]}): #{e.class} #{e.message}"
      begin
        csv.close
      rescue
        nil
      end
      raise
    end

    # At this point we have a payload; write CSV with ID in first column
    pronunciation = extract_pronunciation(row[:furigana])
    pronunciation = extract_pronunciation(row[:jp]) if pronunciation.empty?
    pronunciation = row[:jp].to_s if pronunciation.empty?

    sentence_pronunciation = strip_furigana(payload[:sentence_jp])
    sentence_pronunciation = payload[:sentence_jp].to_s.strip if sentence_pronunciation.empty?

    csv << [
      "JPN1K-#{next_id}",
      next_id,
      row[:jp],
      row[:en],
      row[:furigana],
      pronunciation,
      payload[:sentence_jp],
      payload[:sentence_en],
      payload[:sentence_romaji],
      sentence_pronunciation,
      payload[:explanation],
      "<img src='JPN1K-#{next_id}.jpeg'>"
    ]
    csv.flush
    puts "  ↳ JP: #{payload[:sentence_jp]}"
    puts "  ↳ EN: #{payload[:sentence_en]}"
    puts "  ↳ RM: #{payload[:sentence_romaji]}"
    puts "  ↳ EX: #{payload[:explanation]}" if payload[:explanation].to_s.strip != ""
    next_id += 1

    # Now that we've processed current row, it becomes known for next rows
    cumulative_known << row[:jp]
    cumulative_known << row[:furigana]
  end

  csv.close
  puts "Done. Wrote #{OUTPUT_CSV}"
end

if __FILE__ == $0
  main
end
