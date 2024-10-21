# YouTube Transcript Analyzer

##  Purpose
When it comes to learning new things, I find that watching YouTube videos is one of the most effective ways to do so. Sometimes it might be usefull to get a sense of the main topics that are being discussed. This script provides a convenient way to analyze YouTube video transcripts using LLM, providing a list of main topics discussed in the video with a short description and time stamps. It also provides a way to ask questions about the video content and receive AI-generated answers based on the transcript. On-top of that it provides youtube link with timestamp to the parts of the video where the answer to the question is found.

As some of videos might be lengthy I provide a quick estimate of the input tokens and the cost of the API call.

## Video Demo

![Demo](https://github.com/user-attachments/assets/88ed975b-5c4f-45ac-83f4-33db253f50cf)


## Key Features

1.  **YouTube Transcript API Integration**: The script utilizes the `youtube_transcript_api` to fetch transcripts directly from YouTube videos. This eliminates the need for manual transcript preparation and allows for quick analysis of any accessible YouTube content.

2. **Structured Output with `Instructor`**: I used Instructor library to be able to return a nice table with the main topics of the video.
 
3. **Ability to store transcript in a csv file**: I have noticed that sometimes Youtube blocks the use of API to get the transcripts, so I added a functionality to store the transcript in a csv file and load it later.

4. **Streaming Structural Output**: One of the challenges addressed in this script is streaming structural output. The approach used here involves:
   - Using the `Partial` class from Instructor to handle incomplete data during streaming.
   - Checking for full row population before adding topics to the display table, to avoid flickering table.
   - Continuously updating the live display as new data becomes available.

## Usage

Run the script and follow the prompts to analyze a YouTube video or a local transcript file. If Youtube is blocking your IP address, you can use a VPN to bypass the block or use one of the .csv files to try it out.

