# -*- coding: utf-8 -*-

from chatterbot import ChatBot
import logging

# chatbot = ChatBot('Ron Obvious', trainer='chatterbot.trainers.ChatterBotCorpusTrainer')
# #
# chatbot.train("chatterbot.corpus.english.conversations")
# chatbot.get_response("Hello, how are you today?")



from chatterbot import ChatBot


bot = ChatBot(
    "Jarvis",
    preprocessors=[
        'chatterbot.preprocessors.clean_whitespace',
        'chatterbot.preprocessors.convert_to_ascii'
    ],
    logic_adapters=[
        {
            "import_path": "chatterbot.logic.BestMatch",
            "statement_comparison_function": "chatterbot.comparisons.levenshtein_distance",
            "response_selection_method": "chatterbot.response_selection.get_first_response"
        },
        "chatterbot.logic.MathematicalEvaluation",
        {
            'import_path': "chatterbot.logic.TimeLogicAdapter",
            'threshold': 0.9
        },
        {
            'import_path': 'chatterbot.logic.LowConfidenceAdapter',
            'threshold': 0.65,
            'default_response': 'I am sorry, but I do not understand.'
        }
    ],
    trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
)

bot.train("chatterbot.corpus.english.conversations")

# Train the chat bot with a few responses
# bot.train([
#     'How can I help you?',
#     'I want to create a chat bot',
#     'Have you read the documentation?',
#     'No, I have not',
#     'This should help get you started: http://chatterbot.rtfd.org/en/latest/quickstart.html'
# ])
print bot.get_response("What is 90*10?")
print bot.get_response("What time is it?")
print bot.get_response("How are you doing?")
print bot.get_response('How do I make an omelette?')