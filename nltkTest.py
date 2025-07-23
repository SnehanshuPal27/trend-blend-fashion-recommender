import nltk


nltk.download('punkt_tab')
nltk.data.path.append(r"C:\Users\Snehanshu\nltk_data")


try:
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize("Test sentence for punkt_tab.", language="english")
    print("NLTK punkt_tab works correctly!")
except LookupError as e:
    print(f"Error loading punkt_tab: {e}")
