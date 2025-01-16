from transformers import AutoTokenizer, TFAutoModelForCausalLM
import tensorflow as tf
import time

model_name = "facebook/opt-125m"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Create 2 virtual GPUs with 1GB memory each
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=400), ]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

device_placement = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
DEVICE_OPTIONS = [
    "/CPU:0",
    "/GPU:0",
    "/GPU:1",
]
device_placement = list(map(lambda x: DEVICE_OPTIONS[x], device_placement))


start_time = time.time()
model = TFAutoModelForCausalLM.from_pretrained(model_name, device_placement=device_placement)
print("Time until loading model:", time.time() - start_time)
tokenizer = AutoTokenizer.from_pretrained(model_name, from_tf=True)
print("Time until loading tokenizer:", time.time() - start_time)

inputs = tokenizer("I love Hugging Face!", return_tensors="tf")
print("Time until tokenization:", time.time() - start_time)

start_time = time.time()
outputs = model(**inputs)
print("Time for inference warmup:", time.time() - start_time)

max_length = 50
generated_ids = inputs["input_ids"]

start_time = time.time()
gen_token = 0
for gen_token in range(1, max_length+1):
    outputs = model(input_ids=generated_ids)
    logits = outputs.logits
    predicted_id = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
    generated_ids = tf.concat([generated_ids, tf.expand_dims(predicted_id, axis=-1)], axis=-1)

    # Stop if end-of-sequence token is generated
    if tokenizer.decode(predicted_id.numpy()[0]) == tokenizer.eos_token:
        break

print("Time for generating text:", time.time() - start_time)
# output average time
print("Average time:", (time.time() - start_time) / gen_token)


# Decode the generated sequence
generated_text = tokenizer.decode(generated_ids.numpy()[0], skip_special_tokens=True)
print("Generated text:", generated_text)