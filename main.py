from WordAlignment import WordAlignment

if __name__ == "__main__":
    sentence1 = "Today I went to the supermarket to buy apples".split()
    sentence2 = "Oggi io sono andato al supermercato a comprare le mele".split()
    BERT_NAME = "bert-base-multilingual-cased"
    wa = WordAlignment(BERT_NAME, BERT_NAME)
    _, decoded = wa.get_alignment(sentence1, sentence2, calculate_decode=True)
    for (sentence1_w, sentence2_w) in decoded:
        print(sentence1_w, "\t--->", sentence2_w)

    print("-"*35)

    _, decoded = wa.get_alignment(sentence2, sentence1, calculate_decode=True)
    for (sentence2_w, sentence1_w) in decoded:
        print(sentence2_w, "\t--->", sentence1_w)