from sentence_transformers import CrossEncoder, SentenceTransformer

### Sentence Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_PATH = "models/embedding/"

#### Cross Encoder Model
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_PATH = "models/cross_encoder/"


def get_embedding_model():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_model.save(EMBEDDING_MODEL_PATH)


def get_cross_encoder_model():
    cross_encoder_model = CrossEncoder(CROSS_ENCODER_MODEL)
    cross_encoder_model.save(CROSS_ENCODER_PATH)


def main():

    print("get_embedding_model()")
    get_embedding_model()

    print("get_cross_encoder_model()")
    get_cross_encoder_model()


if __name__ == "__main__":
    main()
