use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Time {
    Seconds(f64),
    Milliseconds(f64),
    Microseconds(f64),
    Nanoseconds(f64),
}

#[derive(Debug, Default)]
pub struct Timer {
    start_time: Option<Instant>,
    last_tick: Option<Instant>,
    elapsed_since_tick: Duration,
    elapsed_total: Duration,
    running: bool,
}

impl Timer {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&mut self) {
        let now = Instant::now();
        self.running = true;
        self.start_time = Some(now);
        self.last_tick = Some(now);
        self.elapsed_since_tick = Duration::ZERO;
        self.elapsed_total = Duration::ZERO;
    }

    pub fn stop(&mut self) {
        if !self.running {
            return;
        }

        let now = Instant::now();
        if let Some(start_time) = self.start_time {
            self.elapsed_total = now - start_time;
        }
        if let Some(last_tick) = self.last_tick {
            self.elapsed_since_tick = now - last_tick;
        }

        self.last_tick = Some(now);
        self.running = false;
    }

    pub fn tick(&mut self) {
        if !self.running {
            return;
        }

        let now = Instant::now();
        if let Some(last_tick) = self.last_tick {
            self.elapsed_since_tick = now - last_tick;
        } else {
            self.elapsed_since_tick = Duration::ZERO;
        }
        if let Some(start_time) = self.start_time {
            self.elapsed_total = now - start_time;
        }
        self.last_tick = Some(now);
    }

    pub fn reset(&mut self) {
        self.elapsed_since_tick = Duration::ZERO;
        self.elapsed_total = Duration::ZERO;

        if self.running {
            let now = Instant::now();
            self.start_time = Some(now);
            self.last_tick = Some(now);
        } else {
            self.start_time = None;
            self.last_tick = None;
        }
    }

    pub fn elapsed<U: TimeUnit>(&self) -> Time {
        duration_to_time::<U>(self.elapsed_since_tick)
    }

    pub fn elapsed_total<U: TimeUnit>(&self) -> Time {
        duration_to_time::<U>(self.total_duration())
    }
}

impl Time {
    pub fn value(self) -> f64 {
        match self {
            Time::Seconds(value) => value,
            Time::Milliseconds(value) => value,
            Time::Microseconds(value) => value,
            Time::Nanoseconds(value) => value,
        }
    }
}

pub trait TimeUnit {
    const MULTIPLIER: f64;
    fn wrap(value: f64) -> Time;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seconds;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Milliseconds;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Microseconds;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Nanoseconds;

impl TimeUnit for Seconds {
    const MULTIPLIER: f64 = 1.0;

    fn wrap(value: f64) -> Time {
        Time::Seconds(value)
    }
}

impl TimeUnit for Milliseconds {
    const MULTIPLIER: f64 = 1_000.0;

    fn wrap(value: f64) -> Time {
        Time::Milliseconds(value)
    }
}

impl TimeUnit for Microseconds {
    const MULTIPLIER: f64 = 1_000_000.0;

    fn wrap(value: f64) -> Time {
        Time::Microseconds(value)
    }
}

impl TimeUnit for Nanoseconds {
    const MULTIPLIER: f64 = 1_000_000_000.0;

    fn wrap(value: f64) -> Time {
        Time::Nanoseconds(value)
    }
}

impl Timer {
    fn total_duration(&self) -> Duration {
        if self.running {
            if let Some(start_time) = self.start_time {
                Instant::now() - start_time
            } else {
                self.elapsed_total
            }
        } else {
            self.elapsed_total
        }
    }
}

#[inline]
fn duration_to_time<U: TimeUnit>(duration: Duration) -> Time {
    let seconds = duration.as_secs_f64();
    U::wrap(seconds * U::MULTIPLIER)
}
