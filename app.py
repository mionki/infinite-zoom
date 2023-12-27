from flask import Flask, jsonify, request
from zoom import zoom

app = Flask(__name__)

@app.route('/generate-video', methods=['POST'])
def generate_video():
    data = request.json
    
    # Extract required parameters from the POST request
    model_id = data.get('model_id')
    prompts_array = data.get('prompts_array')
    negative_prompt = data.get('negative_prompt')
    num_outpainting_steps = data.get('num_outpainting_steps')
    guidance_scale = data.get('guidance_scale')
    num_inference_steps = data.get('num_inference_steps')
    custom_init_image = data.get('custom_init_image')
    
    # Call the zoom_api function with provided parameters
    s3_url = zoom(
        model_id,
        prompts_array,
        negative_prompt,
        num_outpainting_steps,
        guidance_scale,
        num_inference_steps,
        custom_init_image
    )
    
    return jsonify({"s3_url": s3_url})

if __name__ == '__main__':
    app.run(debug=True)

