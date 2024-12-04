import pyttsx3


def text_to_speech(text, rate=200, volume=1.0, voice_id=None):
    """
    Convert text to speech.

    Parameters:
        text (str): The text to be converted to speech.
        rate (int): Speed of speech (default: 200 words per minute).
        volume (float): Volume level (0.0 to 1.0, default: 1.0).
        voice_id (str): Optional voice ID for selecting a male or female voice.
    """
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set properties
    engine.setProperty('rate', rate)  # Set speed of speech
    engine.setProperty('volume', volume)  # Set volume level

    # Set voice if a specific voice is required
    if voice_id is not None:
        voices = engine.getProperty('voices')
        for voice in voices:
            if voice.id == voice_id:
                engine.setProperty('voice', voice.id)
                break

    # Speak the text
    engine.say(text)
    engine.runAndWait()


# Example usage
if __name__ == "__main__":
    text = "Hello, I hope you're having a great day!"
    text_to_speech(text)
