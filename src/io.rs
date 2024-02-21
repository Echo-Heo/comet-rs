#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]

const NUM_PORTS: usize = 256;

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IOC {
    in_pin:   bool,
    out_pin:  bool,
    port:     u16,
    binding:  [u8; NUM_PORTS],
    is_bound: [bool; NUM_PORTS],
    ports:    [u64; NUM_PORTS],
}
impl IOC {
    pub const fn new() -> Self {
        Self {
            in_pin:   false,
            out_pin:  false,
            port:     0,
            binding:  [0; NUM_PORTS],
            is_bound: [false; NUM_PORTS],
            ports:    [0; NUM_PORTS],
        }
    }
}

impl IOC {
    pub fn send_out(&mut self, port: u16, data: u64) {
        let port = port as usize % NUM_PORTS;
        self.out_pin = true;
        self.port = port as u16;
        self.ports[port] = data;
    }
    pub const fn port_data(&self, port: u16) -> u64 { self.ports[port as usize % NUM_PORTS] }
}
