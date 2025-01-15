from transformers import AutoTokenizer, TFAutoModelForCausalLM
import tensorflow as tf
model_name = "facebook/opt-2.7b"

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     # Create 2 virtual GPUs with 1GB memory each
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=8000), ]
#         )
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

device_placement = [0] * 34  # pre-process, 32 layers, and post-process
DEVICE_OPTIONS = [
    "/CPU:0",
    "/GPU:0",
    "/GPU:1",
]
device_placement = list(map(lambda x: DEVICE_OPTIONS[x], device_placement))


with tf.device('/GPU:0'):
    model = TFAutoModelForCausalLM.from_pretrained(model_name, device_placement=device_placement)
    tokenizer = AutoTokenizer.from_pretrained(model_name, from_tf=True)

    inputs = tokenizer("I love Hugging Face!", return_tensors="tf")
    outputs = model(**inputs)

    max_length = 50
    generated_ids = inputs["input_ids"]

    for _ in range(max_length):
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits
        predicted_id = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
        generated_ids = tf.concat([generated_ids, tf.expand_dims(predicted_id, axis=-1)], axis=-1)

        # Stop if end-of-sequence token is generated
        if tokenizer.decode(predicted_id.numpy()[0]) == tokenizer.eos_token:
            break

    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_ids.numpy()[0], skip_special_tokens=True)
    print("Generated text:", generated_text)