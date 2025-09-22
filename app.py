from flask import Flask, render_template, request, jsonify, redirect, url_for
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
from torchvision import transforms
import os
import uuid
from datetime import datetime
import json

try:
    from disease_info import disease_recommendations
except ImportError:

    disease_recommendations = {
        "default": {
            "cause": "Unknown cause",
            "cure": "Consult an agricultural expert"
        }
    }

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
BLOG_UPLOAD_FOLDER = 'static/blog_uploads'
BLOG_DATA_FILE = 'blog_posts.json'  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BLOG_UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BLOG_UPLOAD_FOLDER'] = BLOG_UPLOAD_FOLDER


def load_blog_posts():
    try:
        if os.path.exists(BLOG_DATA_FILE):
            with open(BLOG_DATA_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
  
    return [
        {
            'id': 1,
            'title': 'Dealing with Tomato Blight',
            'excerpt': 'Learn how I successfully managed early blight in my tomato crop using organic methods...',
            'image': 'https://images.unsplash.com/photo-1591637333184-19aa84b3e01f?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80',
            'author': 'Rajesh Kumar',
            'date': 'May 15, 2023',
            'likes': 24,
            'type': 'image'
        },
        {
            'id': 2,
            'title': 'Natural Remedies for Aphids',
            'excerpt': 'Discover effective natural solutions to control aphid infestations without harmful pesticides...',
            'image': 'https://images.unsplash.com/photo-1596627117033-5f985cd23252?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80',
            'author': 'Priya Sharma',
            'date': 'June 2, 2023',
            'likes': 37,
            'type': 'image'
        }
    ]


def save_blog_posts(posts):
    try:
        with open(BLOG_DATA_FILE, 'w') as f:
            json.dump(posts, f)
    except:
        pass


blog_posts = load_blog_posts()


MODEL_DIR = "./plant_disease_model"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval() 
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_extractor = None


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def save_blog_file(file):
    if file and file.filename != '':
       
        file_ext = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(app.config['BLOG_UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
   
        if file_ext in ['.mp4', '.mov', '.avi', '.wmv', '.webm']:
            file_type = 'video'
        else:
            file_type = 'image'
            
        return url_for('static', filename=f'blog_uploads/{unique_filename}'), file_type
    return None, None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html', posts=blog_posts)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('dashboard.html')
    
    if 'image' not in request.files:
        return render_template('dashboard.html', error="No file uploaded")
    
    image = request.files['image']

    if image.filename == '':
        return render_template('dashboard.html', error="No file selected")
    
    if not model:
        return render_template('dashboard.html', error="Model not loaded. Please check the model directory.")
    
    if image:
        # Save uploaded image
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(img_path)

        # Preprocess image
        img_tensor = preprocess_image(img_path)

        # Make prediction
        with torch.no.grad():
            outputs = model(img_tensor)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1) 
            conf, class_idx = torch.max(probs, dim=-1)  
            conf = conf.item()
            class_idx = class_idx.item()

       
        if conf < 0.7: 
            return render_template(
                'prediction.html',
                disease="Uncertain",
                cause="Could not detect properly.",
                cure="Please upload a clear plant image with visible leaves.",
                image_path=img_path,
                confidence=f"{conf:.2f}"
            )

        predicted_class = model.config.id2label[class_idx]

 
        if predicted_class in disease_recommendations:
            cause = disease_recommendations[predicted_class]['cause']
            cure = disease_recommendations[predicted_class]['cure']
        else:
            cause = "No information available"
            cure = "No information available"

        return render_template(
            'prediction.html',
            disease=predicted_class,
            cause=cause,
            cure=cure,
            image_path=img_path,
            confidence=f"{conf:.2f}"
        )

@app.route('/gps', methods=['POST'])
def gps_predict():
    return "GPS-based prediction feature coming soon!"

@app.route('/blog/new', methods=['GET', 'POST'])
def new_blog_post():
    if request.method == 'POST':
       
        title = request.form.get('title')
        content = request.form.get('content')
        author = request.form.get('author')
        image = request.files.get('image')
        

        file_url, file_type = save_blog_file(image)
        

        current_date = datetime.now().strftime("%B %d, %Y")

        new_post = {
            'id': len(blog_posts) + 1,
            'title': title,
            'excerpt': content[:150] + '...' if len(content) > 150 else content,
            'image': file_url if file_url else 'https://images.unsplash.com/photo-1416879595882-3373a0480b5b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80',
            'author': author,
            'date': current_date,
            'likes': 0,
            'content': content, 
            'type': file_type if file_type else 'image'  
        }
        blog_posts.append(new_post)
        save_blog_posts(blog_posts) 
        
        return redirect(url_for('home'))
    
    return render_template('new_post.html')

@app.route('/blog/<int:post_id>')
def blog_detail(post_id):
    # Find the post
    for post in blog_posts:
        if post['id'] == post_id:
            return render_template('blog_detail.html', post=post)
    
    return "Post not found", 404

@app.route('/blog/<int:post_id>/like', methods=['POST'])
def like_post(post_id):
    # Find the post and increment like count
    for post in blog_posts:
        if post['id'] == post_id:
            post['likes'] += 1
            save_blog_posts(blog_posts)  # Save to file
            return jsonify({'success': True, 'likes': post['likes']})
    
    return jsonify({'success': False, 'error': 'Post not found'}), 404

# API endpoint to get updated blog posts
@app.route('/api/blog_posts', methods=['GET'])
def get_blog_posts():
    # Reload from file to ensure we have the latest data
    global blog_posts
    blog_posts = load_blog_posts()
    return jsonify(blog_posts)

if __name__ == '__main__':

    app.run(debug=True)
