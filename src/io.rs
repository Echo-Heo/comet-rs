#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![deny(unsafe_code)]

use std::fmt::Debug;

use crate::{
    comet::Emulator,
    safety::{Interrupt, Port},
};

const NUM_PORTS: usize = 256;

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy)]
enum ICStatus {
    IOCStandBy,
    IOCBindIntWaitingForPort,
    IOCBindIntWaitingForInt,
}

#[derive(Debug, Clone, Copy)]
struct ICData {
    status:   ICStatus,
    bindport: Port,
}
impl ICData {
    const fn new() -> Self {
        Self {
            status:   ICStatus::IOCStandBy,
            bindport: Port(0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum IOCStatus {
    ICStandBy,
    ICWaitingForIvt,
}
impl IOCStatus {
    const fn new() -> Self { Self::ICStandBy }
}

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct IOC {
    pub(crate) in_pin:   bool,
    pub(crate) out_pin:  bool,
    pub(crate) port:     Port,
    pub(crate) binding:  [u8; NUM_PORTS],
    pub(crate) is_bound: [bool; NUM_PORTS],
    pub(crate) ports:    [u64; NUM_PORTS],
    ic_data:             ICData,
    ioc_data:            IOCStatus,
}
impl IOC {
    pub(crate) const fn new() -> Self {
        Self {
            in_pin:   false,
            out_pin:  false,
            port:     Port(0),
            binding:  [0; NUM_PORTS],
            is_bound: [false; NUM_PORTS],
            ports:    [0; NUM_PORTS],
            ic_data:  ICData::new(),
            ioc_data: IOCStatus::new(),
        }
    }
}

#[repr(u16)]
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Ports {
    IC       = 0,
    IOC      = 1,
    MMU      = 2,
    SysTimer = 3,
    TTY      = 10,
}
impl Ports {
    pub(crate) const fn from_port(port: Port) -> Option<Self> {
        match port.0 {
            0 => Some(Self::IC),
            1 => Some(Self::IOC),
            2 => Some(Self::MMU),
            3 => Some(Self::SysTimer),
            10 => Some(Self::TTY),
            _ => None,
        }
    }
    pub(crate) fn run(self, comet: &mut Emulator, data: u64) {
        match self {
            Self::IC => match comet.ioc.ic_data.status {
                ICStatus::IOCStandBy => {
                    if comet.ioc.port_data(Port(Ports::IOC as u16)) == 0 {
                        comet.ioc.ic_data.status = ICStatus::IOCBindIntWaitingForPort;
                    }
                }
                ICStatus::IOCBindIntWaitingForPort => {
                    comet.ioc.ic_data.bindport = Port(data as u16);
                    comet.ioc.ic_data.status = ICStatus::IOCBindIntWaitingForInt;
                }
                ICStatus::IOCBindIntWaitingForInt => {
                    comet.ioc.bind_port(comet.ioc.ic_data.bindport, Interrupt::from_u16(data as u16).unwrap());
                    comet.ioc.ic_data.status = ICStatus::IOCStandBy;
                }
            },
            Self::IOC => match comet.ioc.ioc_data {
                IOCStatus::ICStandBy => {
                    if comet.ioc.port_data(Port(Ports::IOC as u16)) == 0 {
                        comet.ioc.ioc_data = IOCStatus::ICWaitingForIvt;
                    }
                }
                IOCStatus::ICWaitingForIvt => {
                    comet.ic.ivt_base_address = data;
                    comet.ioc.ioc_data = IOCStatus::ICStandBy;
                }
            },
            Self::MMU => todo!("mmu IO"),
            Self::SysTimer => todo!("system timer IO"),
            Self::TTY => {
                if let Some(char) = char::from_u32(data as u32) {
                    print!("{char}");
                }
            }
        }
    }
}

impl IOC {
    pub(crate) fn send_out(&mut self, port: Port, data: u64) {
        let port = Port(port.0 % (NUM_PORTS as u16));
        self.out_pin = true;
        self.port = port;
        self.ports[port.0 as usize] = data;
    }
    pub(crate) const fn port_data(&self, port: Port) -> u64 { self.ports[port.0 as usize % NUM_PORTS] }
    fn bind_port(&mut self, port: Port, interrupt: Interrupt) {
        let port = Port(port.0 % (NUM_PORTS as u16));
        self.is_bound[port.0 as usize] = true;
        self.binding[port.0 as usize] = interrupt as u8;
    }
}
