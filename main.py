import cv2
import numpy as np
import time
import random

import music

last_note_time = 0
song_start = time.time()
SONG_DURATION = 150

lead = music.LeadVoice()
lead_note_time = 0
lead_phrase_duration = 0

PHASES = [
    ("intro",  0,   30,  {"cooldown": 0.0, "volume_mult": 0.4}),
    ("verse",  30,  90,  {"cooldown": 0.0, "volume_mult": 0.5}),
    ("build",  90,  120, {"cooldown": 0.0, "volume_mult": 0.8}),
    ("outro",  120, 150, {"cooldown": 1.5, "volume_mult": 0.3}),
]
TRANSITION_DURATION = 8

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

def processImage(img):
    img = cv2.erode(img, (5,5), iterations = 1)
    img = cv2.dilate(img, (5,5), iterations = 2)

    return img

def getWhiteMask(img):
    lowerBound = np.array([100, 150, 50])
    upperBound = np.array([130, 255, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return processImage(cv2.inRange(hsv, lowerBound, upperBound))

def getOrangeMask(img):
    lowerBound = np.array([0, 0, 0])
    upperBound = np.array([179, 255, 50])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return processImage(cv2.inRange(hsv, lowerBound, upperBound))

def getBlobInfo(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])  # center x
    cy = int(M["m01"] / M["m00"])  # center y

    if area < 300:
        return None
    return cx, cy, area

def getPhaseAndParams(elapsed):
    for i, (phase, start, end, params) in enumerate(PHASES):
        if elapsed < end:
            time_left = end - elapsed
            if time_left < TRANSITION_DURATION and i + 1 < len(PHASES):
                next_phase, _, _, next_params = PHASES[i + 1]
                t = 1.0 - (time_left / TRANSITION_DURATION)
                blended = {
                    "cooldown": params["cooldown"] + t * (next_params["cooldown"] - params["cooldown"]),
                    "volume_mult": params["volume_mult"] + t * (next_params["volume_mult"] - params["volume_mult"]),
                }
                phase = next_phase if t > 0.5 else phase
                return phase, blended
            return phase, params
    return PHASES[-1][0], PHASES[-1][3]

chords_by_position = ["C", "Am", "F", "G"]

def main():
    global last_note_time, lead_note_time, lead_phrase_duration
    cooldown = 0.5
    while True:
        ret, img = cap.read()
        white_mask = getWhiteMask(img)
        orange_mask = getOrangeMask(img)

        img[white_mask > 0] = [255, 100, 0]
        img[orange_mask > 0] = [0, 100, 255]

        elapsed = time.time() - song_start
        now = time.time()
        phase, params = getPhaseAndParams(elapsed)

        cv2.putText(img, phase.upper(), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow('Dis shit', img)
        if cv2.waitKey(1) == ord('q'):
            break
            
        if elapsed >= SONG_DURATION:
            print("Song complete!")
            music.saveRecording(f"track_{int(time.time())}.wav")
            break

        if phase in ("verse", "build"):
            if now > lead_phrase_duration:
                notes, final_note, note_dur = lead.nextPhrase()
                music.playLeadPhraseAsync(notes, final_note, note_dur, pan=0.0, volume=1.0)
                phrase_length = note_dur * len(notes) + (1.0 if note_dur < 0.15 else 1.8)
                rest = random.uniform(0.3, 1.5)
                lead_phrase_duration = now + phrase_length + rest

        if now - last_note_time < cooldown:
            continue

        scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        duration = 0.5 if random.random() > 0.2 else 1.0
        played = False

        white_blob = getBlobInfo(white_mask)
        orange_blob = getBlobInfo(orange_mask)

        if phase == "build":
            if white_blob:
                cx, cy, area = white_blob
                chord_name = chords_by_position[int(cx / 320 * len(chords_by_position))]
                chord = music.chords[chord_name]
                music.playArpeggioAsync(chord, pan=-0.8, volume=params["volume_mult"])
                played = True
            if orange_blob:
                cx, cy, area = orange_blob
                chord_name = chords_by_position[int(cx / 320 * len(chords_by_position))]
                chord = music.chords[chord_name]
                music.playArpeggioAsync(chord, pan=0.8, volume=params["volume_mult"])
                played = True
        elif phase == "intro" or phase == "outro":
            if white_blob:
                cx, cy, area = white_blob
                note = scale[int(cx / 320 * len(scale))]
                volume = min(area / 5000, 1.0) * params["volume_mult"]
                music.playHeldNoteAsync(note, duration=3.0, pan=-0.8, volume=volume)
                played = True
            if orange_blob:
                cx, cy, area = orange_blob
                note = scale[int(cx / 320 * len(scale))]
                volume = min(area / 5000, 1.0) * params["volume_mult"]
                music.playHeldNoteAsync(note, duration=3.0, pan=0.8, volume=volume)
                played = True
        else:
            if white_blob:
                cx, cy, area = white_blob
                note = scale[int(cx / 320 * len(scale))]
                volume = min(area / 5000, 1.0) * params["volume_mult"]
                music.playHeldNoteAsync(note, duration, pan=-0.8, volume=volume)
                played = True
            if orange_blob:
                cx, cy, area = orange_blob
                note = scale[int(cx / 320 * len(scale))]
                volume = min(area / 5000, 1.0) * params["volume_mult"]
                music.playNoteAsync(note, duration, pan=0.8, volume=volume)
                played = True

        if played:
            last_note_time = now
            cooldown = params["cooldown"]
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()