from telethon import TelegramClient
import csv
import os
import emoji


# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, writer, media_dir):
    entity = await client.get_entity(channel_username)
    channel_title = entity.title  # Extract the channel's title
    print(channel_title)
    async for message in client.iter_messages(entity, limit=600): # limit set to 600 for resource optimization but could be upscaled
        media_path = None
        if message.media and hasattr(message.media, 'photo'):
            # Create a unique filename for the photo
            filename = f"{channel_username}_{message.id}.jpg"
            media_path = os.path.join(media_dir, filename)
            # Download the media to the specified directory if it's a photo
            await client.download_media(message.media, media_path)
        
        # Write the channel title along with other data
        clean_message = emoji.replace_emoji(message.message, "") if message.message else '' # Remove emojis from the message
        writer.writerow([channel_title, channel_username, message.id, clean_message, message.date, media_path])
    

async def main(client, custome_channel=None):    
    await client.start()
    
    # Create a directory for media files
    media_dir = '../data/raw/photo'
    os.makedirs(media_dir, exist_ok=True)
 
    # Open the CSV file and prepare the writer
    with open('../data/raw/telegram_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])  # Include channel title in the header
        
        # List of channels to scrape
        channels = [
            '@Shageronlinestore','@Shewabrand', '@helloomarketethiopia', '@Fashiontera', '@nevacomputer'   # Existing channel
                 # You can add more channels here
            
        ]
        
        # Iterate over channels and scrape data into the single CSV file
        for channel in custome_channel  if custome_channel else channels:
            await scrape_channel(client, channel, writer, media_dir)
            print(f"Scraped data from {channel}")
            
async def run_scrapper(client):   
    """
    Main entry point to run the Telegram scrapper.
    This function runs the scrapper in an asynchronous event loop.
    It's designed to be called from a non-async function.
    """
    await main(client)
