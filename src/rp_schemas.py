INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': 'ugly, deformed, disfigured, low quality, blurry, bad anatomy, extra limbs, '
                   'bad hands, missing fingers, watermark, text, signature, worst quality, '
                   'jpeg artifacts, cropped, out of frame'
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1152
    },
    'width': {
        'type': int,
        'required': False,
        'default': 768
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'K_EULER'
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 30
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.0
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.3
    },
    'image_url': {
        'type': str,
        'required': False,
        'default': None
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 4,
        'constraints': lambda img_count: 5 > img_count > 0
    },
    'high_noise_frac': {
        'type': float,
        'required': False,
        'default': None
    },
}
