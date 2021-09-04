import random
"""
function that makes a string that contains a random short story with no more than 30 words of vocabulary, and print it
"""
def make_story():
    #create a list of words that can be used in the story
    words = ["The", "A", "Then", "When", "But", "So", "Although", "Nonetheless", "Consequently", "Hence", "Therefore"]
    #create a list of verbs that can be used in the story
    verbs = ["runs", "walks", "jumps", "skips", "sleeps", "laughs", "cries", "swims", "flies", "sings", "dances"]
    #create a list of nouns that can be used in the story
    nouns = ["cat", "dog", "man", "woman", "boy", "girl", "mountain", "ocean", "forest", "city", "country"]
    #create a list of adjectives that can be used in the story
    adjectives = ["slow", "fast", "loud", "quiet", "wet", "dry", "happy", "sad", "hot", "cold", "soft"]
    #create a list of adverbs that can be used in the story
    adverbs = ["slowly", "fast", "loudly", "quietly", "wetly", "dryly", "happily", "sadly", "hotly", "coldly", "softly"]
    #create a list of prepositions that can be used in the story
    prepositions = ["above", "below", "behind", "beneath", "beside", "between", "near", "over", "under", "above", "across"]
    #create a list of conjunctions that can be used in the story
    conjunctions = ["and", "but", "or", "so", "after", "before", "although", "because", "since", "though", "unless", "until", "when", "where", "whether", "while"]
    #create a list of interjections that can be used in the story
    interjections = ["oh", "ah", "wow", "hooray", "yes", "no", "please", "thank you", "congratulations", "good job", "great", "super"]
    #create a list of articles that can be used in the story
    articles = ["the", "a", "an"]
    #create a list of noun phrases that can be used in the story
    noun_phrases = ["the", "a", "an"]
    #create a list of prepositional phrases that can be used in the story
    prepositional_phrases = ["the", "a", "an"]
    #create a list of verbs that can be used in the story
    verbs = ["runs", "walks", "jumps", "skips", "sleeps", "laughs", "cries", "swims", "flies", "sings", "dances"]
    # make a correct sentence with those words
    sentence = "The " + random.choice(articles) + " " + random.choice(nouns) + " " + random.choice(verbs) + " " + random.choice(adverbs) + "."
    #make a sentence with those words
    sentence2 = "The " + random.choice(articles) + " " + random.choice(nouns) + " " + random.choice(verbs) + " " + random.choice(adverbs) + "."
    #make a sentence with those words
    sentence3 = "The " + random.choice(articles) + " " + random.choice(nouns) + " " + random.choice(verbs) + " " + random.choice(adverbs) + "."
    #make a sentence with those words
    sentence4 = "The " + random.choice(articles) + " " + random.choice(nouns) + " " + random.choice(verbs) + " " + random.choice(adverbs) + "."
    #make a sentence with those words
    sentence5 = "The " + random.choice(articles) + " " + random.choice(nouns) + " " + random.choice(verbs) + " " + random.choice(adverbs) + "."
    #make a sentence with those words
    sentence6 = "The " + random.choice(articles) + " " + random.choice(nouns) + " " + random.choice(verbs) + " " + random.choice(adverbs) + "."
    #make a sentence with those words
    sentence7 = "The " + random.choice(articles) + " " + random.choice(nouns) + " " + random.choice(verbs) + " " + random.choice(adverbs) + "."
    # tell the story
    story = sentence + " " + sentence2 + " " + sentence3 + " " + sentence4 + " " + sentence5 + " " + sentence6 + " " + sentence7
    #print the story
    print(story)

#call the function
make_story()