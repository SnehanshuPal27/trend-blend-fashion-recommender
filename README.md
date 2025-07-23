# TrendBlend: A Two-Stage Semantic Fashion Recommender

**Course:** CS306 - Machine Learning

**Project Submission**

---

### **Live Demo**

**https://trendblend-service-83876934258.asia-south2.run.app**

---

## 1. Project Overview

**TrendBlend** is an advanced, content-based fashion recommendation system built to understand the real meaning behind what a user is looking for. This project moves beyond simple keyword matching to grasp the semantic context of a user's description, allowing it to provide highly relevant and nuanced recommendations from a large product catalog.

The system is built as a complete web application, with a modern user interface and a scalable backend. The entire deployment process is automated using a CI/CD pipeline on Google Cloud, demonstrating a full-stack approach to machine learning systems.

### Problem Statement

Traditional e-commerce search often fails with complex, descriptive queries. A search for "a light blue shirt for a casual summer day" might just find products with "blue" and "shirt," missing the important concepts of "light," "casual," and "summer." Our goal was to build a system that can interpret this kind of human language to deliver much better recommendations.

### Core Objective

To design, implement, and deploy a machine learning system that uses modern Natural Language Processing (NLP) to give users personalized fashion recommendations from free-form text descriptions.

---

## 2. The Machine Learning Pipeline

To provide both fast and accurate recommendations across thousands of products, we designed a **two-stage recommender system**. This architecture is a common pattern in large-scale commercial systems.

The process is broken down into three main parts: offline pre-computation, a fast retrieval stage, and a precise ranking stage.

### Part A: Pre-computation on Kaggle

The most computationally intensive tasks were performed offline using a Kaggle Notebook to take advantage of its free processing power. This step prepares all the necessary data and models for our live application.

* **Dataset:** We used the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle.
* **Process:** A Python script was run on the Kaggle platform to process all 44,000+ product JSON files. For each product, it generated a rich textual description and then used a Sentence-BERT model to create a vector embedding.
* **Output Artifacts:** This pre-computation step generated three critical files which are used by our live application:
    1.  `fashion_data_bert_faiss.pkl`: A serialized file containing a DataFrame of all processed product information.
    2.  `fashion_faiss.index`: A Faiss index containing the vector embeddings for all products.
    3.  `ranker_model.pkl`: The trained LightGBM model for the ranking stage.

### Part B: Stage 1 - Retrieval (Finding Candidates)

The first live stage finds a few hundred *potentially relevant* items from the entire catalog very quickly. The goal here is speed and making sure we don't miss any good options (high recall).

* **Technique:** We use a pre-trained **Sentence-BERT (SBERT)** model (`all-MiniLM-L6-v2`) to convert the user's query into a vector embedding.
* **Why SBERT?** Unlike older methods that count words, SBERT understands the meaning of the whole sentence. It knows that "summer dress" and "sundress" are very similar concepts.
* **Indexing for Speed:** The pre-computed product embeddings are stored in a **Faiss (Facebook AI Similarity Search)** index. Faiss allows us to find the most similar product vectors in the index almost instantly.

### Part C: Stage 2 - Ranking (Finding the Best Matches)

The retrieval stage gives us about 100 possible items. The ranking stage takes this smaller list and re-sorts it carefully to put the absolute best matches at the top. The goal here is high precision.

* **Technique:** We use a gradient-boosted decision tree model, **LightGBM**, as our ranker.
* **Smarter Features:** The ranker makes its decision using more than just text similarity. For each candidate item, it considers:
    1.  **Retrieval Score:** The similarity score from the Faiss search.
    2.  **Product Price:** A very important factor in a user's final choice.
* **How We Trained the Ranker:** Without real user click data, we created our own training set. We took sample products, found the top 100 similar items for each, and labeled the top 10 as "relevant" (1) and the rest as "irrelevant" (0). The LightGBM model was then trained to predict how "relevant" an item is.

---

## 3. Data and Feature Engineering

To get the best results from the SBERT model, we needed to give it high-quality text. We engineered a single `full_text` field for each product to create a rich description.

* **Function:** `process_json_file` in `recommender.py`
* **Process:** Instead of only using the `description` field (which was often missing), this function combines many different attributes into one complete string.
* **Fields Combined:** `productDisplayName`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`, and `usage`.
* **Example `full_text`:** "Palm Tree Girls Sp Jace Sko White Skirts Apparel Bottomwear Skirts White Summer Casual"

This method ensures every product has a detailed text description for the model to understand, which greatly improves the quality of the recommendations.

---

## 4. File Structure and Code Explanation

The project is organized into a backend application, a frontend interface, and deployment configuration files.

| File/Folder                     | Purpose                                                                                                                                                             |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`run.py`** | The main Flask web server file. It defines the API endpoints (`/`, `/results`, `/recommend`) and loads the ML models when the server starts.                             |
| **`recommender.py`** | The core machine learning module. It handles downloading models from cloud storage and contains the `get_recommendations` function which runs the two-stage logic.    |
| **`requirements.txt`** | A list of all the Python libraries the project needs to run, such as Flask, Gunicorn, and Faiss.                                                                    |
| **`templates/`** | This folder holds the HTML files for the website's user interface.                                                                                                  |
| &nbsp;&nbsp;&nbsp;`index.html`   | The homepage with the search bar and input form.                                                                                                                    |
| &nbsp;&nbsp;&nbsp;`results.html` | The page that shows the final recommended products to the user.                                                                                                     |
| **`Dockerfile`** | A set of instructions for building a portable Docker container for our application. It uses Gunicorn as the production-grade web server.                              |
| **`.github/workflows/deploy.yml`** | This file defines the automated CI/CD pipeline using GitHub Actions. It automatically builds, tests, and deploys the application whenever code is pushed to `main`. |

---

## 5. Deployment and DevOps

This project was built with professional deployment practices in mind, using a fully automated pipeline.

* **Containerization:** The application is packaged into a **Docker** container. This creates a consistent and isolated environment, ensuring that the application runs the same way on a local machine as it does in the cloud.
* **CI/CD:** We use **GitHub Actions** to automate our entire deployment. Every time we push code to the `main` branch, the workflow automatically:
    1.  Authenticates securely to Google Cloud.
    2.  Builds the Docker image.
    3.  Pushes the image to **Google Artifact Registry**.
    4.  Deploys the new image to **Google Cloud Run**.
    5.  Cleans up old container images to save on storage costs.
* **Scalable Hosting:** The application is hosted on **Google Cloud Run**, a serverless platform that automatically scales based on user traffic. We keep a minimum of one container running at all times to ensure the app is always "warm" and responsive, avoiding slow startup times.
* **Artifact Management:** Large model files are stored in **Google Cloud Storage (GCS)**, separate from our source code. The application securely downloads these models when it starts up.

---

## 6. How to Run the Project

### Local Development (Using Docker)

1.  **Prerequisites:** You need Docker Desktop installed and running.
2.  **Setup:**
    * Place your three model files (`.pkl`, `.index`) in a `/models` folder.
    * Place your GCP Service Account key in the project's root directory as `gcp-key.json`.
    * Create a `docker-compose.yml` file to manage the container setup.
3.  **Run:** From your terminal, run the command `docker-compose up --build`.
4.  **Access:** Open your web browser to `http://localhost:5000`.

### Cloud Deployment

1.  **Prerequisites:** A Google Cloud project with billing enabled.
2.  **Setup:**
    * Create a GCS bucket and upload your model files.
    * Create an Artifact Registry repository for your Docker images.
    * Create a Service Account with the necessary permissions.
    * Add your project ID and service account key as secrets in your GitHub repository settings.
3.  **Deploy:** Simply push your code to the `main` branch of your GitHub repository. The automated workflow will handle the rest.

