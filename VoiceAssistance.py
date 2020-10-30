import speech_recognition as sr
import pyttsx3
import time
import datetime


def greet():
    hour = datetime.datetime.now().hour
    if 6 <= hour < 12:
        speak("Good Morning Marina!")
        print("Good Morning Marina!")
    elif 12 <= hour < 21:
        speak("Good Afternoon Marina!")
        print("Good Afternoon Marina!")
    else:
        speak("Good Evening Marina!")
        print("Good Evening Marina!")


def take_command():
    recognizer = sr.Recognizer()
    recognizer.adjust_for_ambient_noise
    microphone = sr.Microphone()
    with microphone as source:
        print("Please speak...")
        audio = recognizer.listen(source)
        try:
            #statement = recognizer.recognize_google(audio, language='el-GR')
            statement = recognizer.recognize_google(audio, language='en-US')
            print(f"You said: {statement} \n")

        except Exception as e:
            speak("Please repeat")
            print("Please repeat")
            return "None"
        return statement


def speak(text):
    engine.say(text)
    engine.runAndWait()


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', 'voices[1].id')

greet()

print('''Welcome, I am your personal assistant. I can give you data analysis results and geo-visualization.
       Please say options for available commands''')
speak('''Welcome, I am your personal assistant. I can give you data analysis results and geo-visualization.
       Please say options for available commands''')


if __name__ == '__main__':

    while True:
        command = take_command().lower()
        if command == 0:
            continue
        if "options" in command:
            speak("You can select keywords for GitHub repositories,  the time frame, "
                  "the type of map for the results and export them to csv format")
        elif "exit" in command or "No bye" in command or "No thanks" in command:
            speak('Ok, exiting, Good bye')
            break
        elif "keywords" in command or "Github" in command:
            speak('Available keywords for GitHub repositories are:'
                  'Diabetes, Machine learning')
        elif "map" in command or "map type" in command:
            speak('Available map types are:'
                  'Marks clustering, heatmap')
        elif "time" in command or "time frame" in command:
            speak('Available time frames are '
                  'Specific dates,  range of dates')
        elif "specific" in command or "specific dates" in command:
            speak('Please give a specific date')
        #speak("Do you need something else?")
time.sleep(5)


