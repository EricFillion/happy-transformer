from happy_transformer.happy_bert import HappyBERT


def main():
    """testing"""
    happy_bert = HappyBERT()
    token, probs = happy_bert.predict_mask("Who was Jim Henson? Jim [MASK] was a puppeteer.")
    print(token)
    print(probs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Aborting')
