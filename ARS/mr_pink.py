import pvporcupine
import pyaudio
import struct

def detect_wake_word():
    porcupine = pvporcupine.create(
        access_key='6CHlILWC+M38HfFXum2nSsvMtD3RT33ERy0o/5xmTIgzRYqqiM6sfg==',
        keyword_paths=['models/mr-pink.ppn']
    )

    p = pyaudio.PyAudio()
    stream = p.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("Listening for wake word...")

    while True:
        try:
            pcm = stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print("Wake word detected!")
                # Trigger your ARS response here
                break

        except KeyboardInterrupt:
            print("Exiting...")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    porcupine.delete()
