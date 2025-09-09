from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World", "status": "Container is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
