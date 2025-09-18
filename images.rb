#!/usr/bin/env ruby

# Generate an image for each flashcard row in flashcards.csv
# - Reads rows from OUTPUT_PATH CSV (built by build.rb)
# - For each row, asks GPT-5 to craft a concise image prompt
#   based on the English example sentence
# - Sends the prompt to the OpenAI images API to render a jpeg
# - Saves images to ./images with filename equal to the row ID (e.g., JPN1K-18.jpeg)

require "dotenv"
require "openai"
require "csv"
require "json"
require "fileutils"
require "base64"
require "net/http"
require "uri"

Dotenv.load

# Configuration
MODEL = ENV.fetch("OPENAI_MODEL", "gpt-5")
OPENAI_URI_BASE = ENV.fetch("OPENAI_URI_BASE", "https://api.openai.com/v1")

INPUT_CSV = ENV.fetch("OUTPUT_PATH") # same file produced by build.rb
IMAGES_DIR = ENV.fetch("IMAGES_DIR", "images")

START_INDEX = (ENV["START_INDEX"] || "1").to_i # 1-based row index in INPUT_CSV (excluding header)
LIMIT = ENV["LIMIT"]&.to_i # optional limit of rows to process

DEFAULT_CONCURRENCY = 5
CONCURRENCY = begin
  raw = ENV["CONCURRENCY"]
  value = raw.to_i
  value.positive? ? value : DEFAULT_CONCURRENCY
end

LOG_MUTEX = Mutex.new

# Replicate configuration (for Seedream 4)
REPLICATE_API_TOKEN = ENV["REPLICATE_API_TOKEN"]
REPLICATE_MODEL = ENV.fetch("REPLICATE_MODEL", "bytedance/seedream-4")
REPLICATE_ASPECT_RATIO = ENV.fetch("REPLICATE_ASPECT_RATIO", "4:3")

def configure_openai!
  OpenAI.configure do |config|
    config.access_token = ENV.fetch("OPENAI_ACCESS_TOKEN")
    config.uri_base = OPENAI_URI_BASE
    config.log_errors = true
    config.uri_base = ENV["OPENAI_URI_BASE"] if ENV["OPENAI_URI_BASE"]
  end
end

def client
  @client ||= OpenAI::Client.new
end

def developer_message
  image_style = [
    Array.new(5, "Candid (unposed) photography"),
    Array.new(5, "iPhone middayphotography"),
    Array.new(5, "Cinematic"),
    "1990s leica film photography",
    "Watercolor painting",
    "Oil painting",
    "Ink drawing",
    "Pixel art",
    "Cyberpunk",
    "Isometric illustration",
    "Retro-futurist oil painting",
    "Spontaneous smartphone photo",
    "3d high quality video game",
    "High quality anime"
  ].flatten.sample

  gender = [
    "man",
    "woman"
  ].sample

  <<~DEV
    You are an assistant that generates prompts to be sent to an AI image generator based on the text that is sent by the user.
    Your goal is to figure out what image would best illustrate the example sentence sent by the user. If the sentence is quite generic, fabricate a setup that you think will help with the retention. Be creative. There should be a story behind the story your prompt is describing.

    Only respond with the prompt, nothing else.
    Use the following style: #{image_style}
    Maximum 3 sentences.
    If a photo, never during the golden hour.
    If the image is supposed to contain a single person and the gender is not specified, make it a #{gender}. If the sentence specifies a gender, follow it.
    You shouldn't write the message sent by the user on the image itself. You should only illustrate the sentence.
    You should never ask to have any text, date, or time on the image.
    Do not make the image depressing.
    #{[nil, "Do not have them smile unless relevant for the sentence."].sample}
    Avoid showing phones or watches.
  DEV
end

def extract_text_from_response(response)
  # Mirror parsing approach from build.rb for Responses API
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
  text_chunks.join("\n").strip
end

def prompt_for_image(en_sentence)
  input = [
    {role: "developer", content: developer_message},
    {role: "user", content: en_sentence.to_s}
  ]

  response = client.responses.create(parameters: {
    model: MODEL,
    input: input,
    reasoning: {effort: "low"}
  })

  if response.is_a?(Hash) && response["error"]
    err = response["error"]
    message = err.is_a?(Hash) ? (err["message"] || err.inspect) : err.to_s
    raise "OpenAI API error (prompt): #{message}"
  end

  prompt = extract_text_from_response(response)
  raise "Empty prompt from OpenAI for sentence: #{en_sentence.inspect}" if prompt.nil? || prompt.strip.empty?
  prompt.strip
end

def generate_image_jpeg!(prompt)
  # Use Replicate Seedream to render the image and return raw JPEG/PNG bytes
  prediction = replicate_post_prediction!(prompt)
  status = prediction["status"].to_s
  if status == "failed" || status == "canceled"
    raise "Replicate prediction #{status}"
  end

  output = prediction["output"]
  url = extract_image_url_from_output(output)
  raise "Replicate response missing image URL in output" if url.to_s.strip.empty?

  download_bytes_with_redirects(url)
end

def replicate_post_prediction!(prompt)
  raise "Missing REPLICATE_API_TOKEN" if REPLICATE_API_TOKEN.to_s.strip.empty?

  uri = URI("https://api.replicate.com/v1/models/#{REPLICATE_MODEL}/predictions")
  headers = {
    "Authorization" => "Bearer #{REPLICATE_API_TOKEN}",
    "Content-Type" => "application/json",
    "Prefer" => "wait"
  }

  body = {
    input: {
      prompt: prompt,
      aspect_ratio: REPLICATE_ASPECT_RATIO
    }
  }

  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true
  req = Net::HTTP::Post.new(uri.request_uri, headers)
  req.body = JSON.dump(body)

  res = http.request(req)
  unless res.is_a?(Net::HTTPSuccess)
    raise "Replicate API error: HTTP #{res.code} #{res.message} - #{res.body.to_s[0, 500]}"
  end

  json = JSON.parse(res.body)
  if json.is_a?(Hash) && json["error"]
    err = json["error"]
    message = err.is_a?(Hash) ? (err["message"] || err.inspect) : err.to_s
    raise "Replicate API error: #{message}"
  end
  json
end

def extract_image_url_from_output(output)
  # Common Replicate shape: output is a string URL or an array of URLs
  if output.is_a?(String)
    return output if output.start_with?("http://", "https://")
  elsif output.is_a?(Array)
    first_url = output.find { |v| v.is_a?(String) && v.start_with?("http://", "https://") }
    return first_url if first_url
  elsif output.is_a?(Hash)
    # Fallback: search hash values for URLs
    output.values.each do |v|
      if v.is_a?(String) && v.start_with?("http://", "https://")
        return v
      elsif v.is_a?(Array)
        candidate = v.find { |x| x.is_a?(String) && x.start_with?("http://", "https://") }
        return candidate if candidate
      end
    end
  end
  nil
end

def download_bytes_with_redirects(url, limit = 3)
  raise "Too many HTTP redirects" if limit <= 0
  uri = URI(url)
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = (uri.scheme == "https")
  req = Net::HTTP::Get.new(uri.request_uri)
  res = http.request(req)

  case res
  when Net::HTTPSuccess
    res.body
  when Net::HTTPRedirection
    location = res["location"]
    raise "Redirect missing location" if location.to_s.strip.empty?
    download_bytes_with_redirects(location, limit - 1)
  else
    raise "Image download error: HTTP #{res.code} #{res.message}"
  end
end

def ensure_images_dir!
  FileUtils.mkdir_p(IMAGES_DIR)
end

def filepath_for_id(id)
  File.join(IMAGES_DIR, "#{id}.jpeg")
end

def emit_logs(log_events)
  Array(log_events).each do |event|
    level, message = event
    next if message.to_s.empty?

    LOG_MUTEX.synchronize do
      case level
      when :warn
        warn message
      else
        puts message
      end
    end
  end
end

def perform_task(job)
  i, total_rows, id, en_sentence, out_path = job
  logs = []

  begin
    logs << [:info, "[#{i}/#{total_rows}] #{id} → building prompt for: #{en_sentence}"]
    prompt = prompt_for_image(en_sentence)
    logs << [:info, "  ↳ prompt: #{prompt}"]
    jpeg_bytes = generate_image_jpeg!(prompt)
    File.binwrite(out_path, jpeg_bytes)
    logs << [:info, "  ↳ saved #{out_path}"]

    {status: :ok, logs: logs}
  rescue => e
    error_message = "Error for row #{i} (#{id}): #{e.class} #{e.message}"
    logs << [:warn, error_message]

    {
      status: :error,
      logs: logs,
      error_message: error_message,
      backtrace: e.backtrace,
      exception: e
    }
  end
end

def process_tasks_concurrently(tasks, concurrency_limit)
  job_queue = Queue.new
  tasks.each { |task| job_queue << task }
  concurrency_limit.times { job_queue << :__END__ }

  error_mutex = Mutex.new
  first_error = nil

  workers = concurrency_limit.times.map do
    Thread.new do
      loop do
        job = job_queue.pop
        break if job == :__END__

        result = perform_task(job)
        emit_logs(result[:logs])

        if result[:status] == :error
          error_mutex.synchronize do
            first_error ||= result
          end
        end
      end
    end
  end

  workers.each(&:join)

  return unless first_error

  error_message = first_error[:error_message] || first_error[:exception]&.message || "Image generation failed"
  exception = RuntimeError.new(error_message)
  backtrace = first_error[:backtrace] || first_error[:exception]&.backtrace
  exception.set_backtrace(backtrace) if backtrace
  raise exception
end

def main
  configure_openai!
  ensure_images_dir!

  unless File.exist?(INPUT_CSV)
    warn "CSV not found at #{INPUT_CSV}"
    exit 1
  end

  total_rows = 0
  CSV.foreach(INPUT_CSV, headers: true) { total_rows += 1 }

  # Determine window
  start_idx = [START_INDEX, 1].max
  last_idx = if LIMIT && LIMIT > 0
    [start_idx + LIMIT - 1, total_rows].min
  else
    total_rows
  end

  puts "Generating images for rows #{start_idx}..#{last_idx} of #{total_rows}"

  i = 0
  tasks = []
  CSV.foreach(INPUT_CSV, headers: true) do |row|
    i += 1
    next if i < start_idx
    break if i > last_idx

    id = row["ID"].to_s.strip
    en_sentence = row["Sentence EN"].to_s.strip

    if id.empty? || en_sentence.empty?
      warn "Skipping row #{i}: missing ID or Sentence EN"
      next
    end

    out_path = filepath_for_id(id)
    if File.exist?(out_path)
      puts "[#{i}/#{total_rows}] #{id} → exists, skipping (#{out_path})"
      next
    end

    tasks << [
      i,
      total_rows,
      id.dup,
      en_sentence.dup,
      out_path.dup
    ]
  end

  unless tasks.empty?
    concurrency_limit = [CONCURRENCY, tasks.length].min
    puts "Using concurrency #{concurrency_limit} for #{tasks.length} image(s)"
    process_tasks_concurrently(tasks, concurrency_limit)
  end

  puts "Done. Images in #{IMAGES_DIR}"
end

if __FILE__ == $0
  main
end
