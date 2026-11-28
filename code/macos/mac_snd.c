/*
===========================================================================
Metal port: SNDDMA_* on top of the Swift audio output (Audio.swift).

The mixer writes 16-bit interleaved samples into a ring buffer (dma.buffer);
an AVAudioEngine source node drains it and advances the read position that
SNDDMA_GetDMAPos reports.
===========================================================================
*/
#include "../game/q_shared.h"
#include "../client/snd_local.h"
#include "mtl_swift.h"

static qboolean snd_inited;
static volatile int32_t dmapos;     // in samples

qboolean SNDDMA_Init( void ) {
	cvar_t *rate;
	int samples;

	if ( snd_inited ) {
		return qtrue;
	}

	rate = Cvar_Get( "s_macRate", "22050", CVAR_ARCHIVE | CVAR_LATCH );

	// about 1/4 second, a power of two
	dma.speed = rate->integer >= 11025 ? rate->integer : 22050;
	dma.channels = 2;
	dma.samplebits = 16;
	samples = 1;
	while ( samples < dma.speed * dma.channels / 4 ) {
		samples <<= 1;
	}
	dma.samples = samples;
	dma.submission_chunk = 1;
	dma.buffer = calloc( 1, dma.samples * 2 );
	dmapos = 0;

	if ( !MacAudio_Start( (int16_t *)dma.buffer, dma.samples, dma.channels, dma.speed, &dmapos ) ) {
		Com_Printf( "Audio output could not be started\n" );
		free( dma.buffer );
		dma.buffer = NULL;
		return qfalse;
	}
	Com_Printf( "Audio: %d Hz, %d channels, 16-bit, %d sample ring buffer\n", dma.speed, dma.channels, dma.samples );

	snd_inited = qtrue;
	return qtrue;
}

int SNDDMA_GetDMAPos( void ) {
	return snd_inited ? dmapos : 0;
}

void SNDDMA_Shutdown( void ) {
	if ( snd_inited ) {
		MacAudio_Stop();
		snd_inited = qfalse;
	}
	if ( dma.buffer ) {
		free( dma.buffer );
		dma.buffer = NULL;
	}
}

void SNDDMA_BeginPainting( void ) {}
void SNDDMA_Submit( void ) {}

// declared in q_shared.h on Linux and macOS (old glibc memset workaround)
void Snd_Memset( void *dest, const int val, const size_t count ) {
	Com_Memset( dest, val, count );
}
