#![warn(clippy::pedantic)]
#![deny(unsafe_code)]


#[derive(Debug, Clone, Copy)]
pub(crate) struct IntQueueEntry {
    pub(crate) code: u8,
}
impl IntQueueEntry {
    pub(crate) const NEW: Self = IntQueueEntry { code: 0 };
}

#[allow(clippy::upper_case_acronyms)]
#[repr(C)]
#[derive(Debug, Clone)]
pub(crate) struct IC {
    pub(crate) ivt_base_address: u64,
    pub(crate) ret_addr:         u64,
    pub(crate) ret_status:       u64,
    pub(crate) queue:            Vec<IntQueueEntry>,
}
impl IC {
    pub(crate) fn new() -> Self {
        Self {
            ivt_base_address: 0,
            ret_addr:         0,
            ret_status:       0,
            queue:            vec![IntQueueEntry::NEW; 32],
        }
    }
}
