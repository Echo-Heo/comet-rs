#![warn(clippy::pedantic)]
#![deny(unsafe_code)]

use crate::safety::{BranchCond, FloatCastType, FloatPrecision, Instruction, Interrupt, LiType, Nibble, Port, Register, B, E, F, M, R};

/// enum for opcode.
    /// Registers for reading are resolved during the conversion,
    /// So I can merge -r and -i ones.
    /// Proooooobably gets optimized away during compilation
    #[derive(Debug, Clone, Copy)]
    #[rustfmt::skip]
    pub(crate) enum Opcode {
        // System Control
        /// `trigger interrupt imm (see Interrupts)`
        Int { imm: Interrupt },
        /// `return from interrupt`
        Iret,
        /// `resolve interrupt`
        Ires,
        /// `enter user mode and jump to address in rd`
        Usr { rd: Register },

        // Input & Output
        /// `output data in rs to port rd`
        Outr { rd: Register, rs: Register },
        /// `output data in rs to port imm`
        Outi { imm: Port, rs: Register},
        /// `read data from port rs to rd`
        Inr { rd: Register, rs: Register },
        /// `read data from port imm to rd`
        Ini { rd: Register, imm: Port },

        // Control Flow
        /// `push ip, ip ← rs + 4 × imm`
        Jal { rs: Register, imm: u16 },
        /// `rd ← ip, ip ← rs + 4 × imm`
        Jalr { rd: Register, rs: Register, imm: u16 },
        /// `pop ip`
        Ret,
        /// `ip ← rs`
        Retr { rs: Register },
        /// `ip ← pc + 4 × imm`
        B { cc: BranchCond, imm: u32 },

        // Stack Operations
        /// `sp ← sp - 8, mem[sp] ← rs`
        Push { rs: Register },
        /// `rd ← mem[sp], sp ← sp + 8`
        Pop { rd: Register },
        /// `push fp, fp = sp; enter stack frame`
        Enter,
        /// `sp = fp, pop fp; leave stack frame`
        Leave,

        // Data Flow
        /// See `LiType`
        Li { rd: Register, func: LiType, imm: u16 },
        /// `rd ← mem[rs + (i64)off + rn << sh]`
        Lw { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },
        /// `rd[31..0] ← mem[rs + (i64)off + rn << sh]`
        Lh { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },
        /// `rd ← mem[rs + (i64)off + rn << sh]`
        Lhs { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },
        /// `rd[15..0] ← mem[rs + (i64)off + rn << sh]`
        Lq { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },
        /// `rd ← mem[rs + (i64)off + rn << sh]`
        Lqs { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },
        /// `rd[7..0] ← mem[rs + (i64)off + rn << sh]`
        Lb { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },
        /// `rd ← mem[rs + (i64)off + rn << sh]`
        Lbs { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },
        /// `mem[rs + off + rs << sh] ← (i64)rd`
        Sw { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },
        /// `mem[rs + off + rs << sh] ← (i32)rd`
        Sh { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },
        /// `mem[rs + off + rs << sh] ← (i16)rd`
        Sq { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },
        /// `mem[rs + off + rs << sh] ← (i8)rd`
        Sb { rd: Register, rs: Register, rn: Register, sh: Nibble, off: u8 },

        // Comparisons
        /// `compare and set flags`
        Cmpr { r1: Register, r2: Register },
        /// `compare and set flags. if the immediate value is first, s is set to 1`
        Cmpi { r1: Register, s: bool, imm: u16 },

        // Arithmetic Operations
        /// `rd ← r1 + r2`
        Addr { rd: Register, r1: Register, r2: Register },
        /// `rd ← r1 + (i64)imm`
        Addi { rd: Register, r1: Register, imm: u16 },
        /// `rd ← r1 - r2`
        Subr { rd: Register, r1: Register, r2: Register },
        /// `rd ← r1 - (i64)imm`
        Subi { rd: Register, r1: Register, imm: u16 },
        /// `rd ← r1 × r2 (signed)`
        Imulr { rd: Register, r1: Register, r2: Register },
        /// `rd ← r1 × (i64)imm (signed)`
        Imuli { rd: Register, r1: Register, imm: u16 },
        /// `rd ← r1 ÷ r2 (signed)`
        Idivr { rd: Register, r1: Register, r2: Register },
        /// `rd ← r1 ÷ (i64)imm (signed)`
        Idivi { rd: Register, r1: Register, imm: u16 },
        /// `rd ← r1 × r2 (unsigned)`
        Umulr { rd: Register, r1: Register, r2: Register },
        /// `rd ← r1 × (u64)imm (unsigned)`
        Umuli { rd: Register, r1: Register, imm: u16 },
        /// `rd ← r1 ÷ r2 (unsigned)`
        Udivr { rd: Register, r1: Register, r2: Register },
        /// `rd ← r1 ÷ (u64)imm (unsigned)`
        Udivi { rd: Register, r1: Register, imm: u16 },
        /// `rd ← rem(r1, r2)`
        Remr { rd: Register, r1: Register, r2: Register },
        /// `rd ← rem(r1, i64(imm))`
        Remi { rd: Register, r1: Register, imm: u16 },
        /// `rd ← mod(r1, r2)`
        Modr { rd: Register, r1: Register, r2: Register },
        /// `rd ← mod(r1, i64(imm))`
        Modi { rd: Register, r1: Register, imm: u16 },

        // Bitwise Operations
        /// `rd ← r1 & r2`
        Andr { rd: Register, r1: Register, r2: Register },
        /// `rd ← r1 & (u64)imm`
        Andi { rd: Register, r1: Register, imm: u16 },
        /// `rd ← r1 | r2`
        Orr { rd: Register, r1: Register, r2: Register },
        /// `rd ← r1 | (u64)imm`
        Ori { rd: Register, r1: Register, imm: u16 },
        /// `rd ← !(r1 | r2)`
        Norr { rd: Register, r1: Register, r2: Register },
        /// `rd ← !(r1 | (u64)imm)`
        Nori { rd: Register, r1: Register, imm: u16 },
        /// `rd ← r1 ^ r2`
        Xorr { rd: Register, r1: Register, r2: Register },
        /// `rd ← r1 ^ (u64)imm`
        Xori { rd: Register, r1: Register, imm: u16 },
        /// `rd ← r1 << r2`
        Shlr { rd: Register, r1: Register, r2: Register },
        /// `rd ← r1 << (u64)imm`
        Shli { rd: Register, r1: Register, imm: u16 },
        /// `rd ← (i64)r1 >> r2`
        Asrr { rd: Register, r1: Register, r2: Register },
        /// `rd ← (i64)r1 >> (u64)imm`
        Asri { rd: Register, r1: Register, imm: u16 },
        /// `rd ← (u64)r1 >> r2`
        Lsrr { rd: Register, r1: Register, r2: Register },
        /// `rd ← (u64)r1 >> (u64)imm`
        Lsri { rd: Register, r1: Register, imm: u16 },
        /// `rd ← if r2 in 0..64 { r1[r2] } else { 0 }`
        Bitr { rd: Register, r1: Register, r2: Register },
        /// `rd ← if imm in 0..64 { r1[imm] } else { 0 }`
        Biti { rd: Register, r1: Register, imm: u16 },

        // Floating-Point Operations
        /// `rd ← comp(r1, r2)`
        Fcmp { r1: Register, r2: Register, p: FloatPrecision },
        /// `rd ← (f)rs`
        Fto { rd: Register, rs: Register, p: FloatPrecision },
        /// `rd ← (i64)rs`
        Ffrom { rd: Register, rs: Register, p: FloatPrecision },
        /// `rd ← -rs`
        Fneg { rd: Register, rs: Register, p: FloatPrecision },
        /// `rd ← |rs|`
        Fabs { rd: Register, rs: Register, p: FloatPrecision },
        /// `rd ← r1 + r2`
        Fadd { rd: Register, r1: Register, r2: Register, p: FloatPrecision },
        /// `rd ← r1 - r2`
        Fsub { rd: Register, r1: Register, r2: Register, p: FloatPrecision },
        /// `rd ← r1 × r2`
        Fmul { rd: Register, r1: Register, r2: Register, p: FloatPrecision },
        /// `rd ← r1 ÷ r2`
        Fdiv { rd: Register, r1: Register, r2: Register, p: FloatPrecision },
        /// `rd +← r1 × r2`
        Fma { rd: Register, r1: Register, r2: Register, p: FloatPrecision },
        /// `rd ← √r1`
        Fsqrt { rd: Register, r1: Register, p: FloatPrecision },
        /// `rd ← min(r1, r2)`
        Fmin { rd: Register, r1: Register, r2: Register, p: FloatPrecision },
        /// `rd ← max(r1, r2)`
        Fmax { rd: Register, r1: Register, r2: Register, p: FloatPrecision },
        /// `rd ← ceil(r1)`
        Fsat { rd: Register, r1: Register, p: FloatPrecision },
        /// `rd ← cast(r1)`
        Fcnv { rd: Register, r1: Register, p: FloatCastType },
        /// `rd ← isnan(r1)`
        Fnan { rd: Register, r1: Register, p: FloatPrecision },
    }

macro_rules! impl_opcode {
        (impl Opcode {
            $($NAME: ident = $value: expr;)*
        }) => {
            impl Opcode {
                $(pub(crate) const $NAME: u8 = $value;)*
            }
        };
    }

impl_opcode! {
    impl Opcode {
        // System Control
        CTRL  = 0x01;

        // Input & Output
        OUTR  = 0x02;
        OUTI  = 0x03;
        INR   = 0x04;
        INI   = 0x05;

        // Control Flow
        JAL   = 0x06;
        JALR  = 0x07;
        RET   = 0x08;
        RETR  = 0x09;
        BCC   = 0x0A;

        // Stack Operations
        PUSH  = 0x0B;
        POP   = 0x0C;
        ENTER = 0x0D;
        LEAVE = 0x0E;

        // Data Flow
        LI    = 0x10;
        LW    = 0x11;
        LH    = 0x12;
        LHS   = 0x13;
        LQ    = 0x14;
        LQS   = 0x15;
        LB    = 0x16;
        LBS   = 0x17;
        SW    = 0x18;
        SH    = 0x19;
        SQ    = 0x1A;
        SB    = 0x1B;

        // Comparisons
        CMPR  = 0x1E;
        CMPI  = 0x1F;

        // Arithmetic Operations
        ADDR  = 0x20;
        ADDI  = 0x21;
        SUBR  = 0x22;
        SUBI  = 0x23;
        IMULR = 0x24;
        IMULI = 0x25;
        IDIVR = 0x26;
        IDIVI = 0x27;
        UMULR = 0x28;
        UMULI = 0x29;
        UDIVR = 0x2A;
        UDIVI = 0x2B;
        REMR  = 0x2C;
        REMI  = 0x2D;
        MODR  = 0x2E;
        MODI  = 0x2F;

        // Bitwise Operations
        ANDR  = 0x30;
        ANDI  = 0x31;
        ORR   = 0x32;
        ORI   = 0x33;
        NORR  = 0x34;
        NORI  = 0x35;
        XORR  = 0x36;
        XORI  = 0x37;
        SHLR  = 0x38;
        SHLI  = 0x39;
        ASRR  = 0x3A;
        ASRI  = 0x3B;
        LSRR  = 0x3C;
        LSRI  = 0x3D;
        BITR  = 0x3E;
        BITI  = 0x3F;

        // Floating-Point Operations
        FCMP  = 0x40;
        FTO   = 0x41;
        FFROM = 0x42;
        FNEG  = 0x43;
        FABS  = 0x44;
        FADD  = 0x45;
        FSUB  = 0x46;
        FMUL  = 0x47;
        FDIV  = 0x48;
        FMA   = 0x49;
        FSQRT = 0x4A;
        FMIN  = 0x4B;
        FMAX  = 0x4C;
        FSAT  = 0x4D;
        FCNV  = 0x4E;
        FNAN  = 0x4F;
    }
}
impl Opcode {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn from_instruction(i: Instruction) -> Result<Self, Interrupt> {
        match i.opcode() {
            // System Control
            Self::CTRL => {
                let F { imm, func, rd } = i.f();
                match func {
                    Nibble(0) => Ok(Self::Int {
                        imm: Interrupt::from_u16(imm)?,
                    }),
                    Nibble(1) => Ok(Self::Iret),
                    Nibble(2) => Ok(Self::Ires),
                    Nibble(3) => Ok(Self::Usr { rd }),
                    _ => Err(Interrupt::InvalidInstruction),
                }
            }
            // Input & Output
            opcode @ Self::OUTR..=Self::INI => {
                let M { imm, rs, rd } = i.m();
                match opcode {
                    Self::OUTR => Ok(Self::Outr { rd, rs }),
                    Self::OUTI => Ok(Self::Outi { imm: Port(imm), rs }),
                    Self::INR => Ok(Self::Inr { rd, rs }),
                    Self::INI => Ok(Self::Ini { rd, imm: Port(imm) }),
                    _ => unreachable!(),
                }
            }
            // Control Flow
            opcode @ Self::JAL..=Self::RETR => {
                let M { imm, rs, rd } = i.m();
                match opcode {
                    Self::JAL => Ok(Self::Jal { rs, imm }),
                    Self::JALR => Ok(Self::Jalr { rd, rs, imm }),
                    Self::RET => Ok(Self::Ret),
                    Self::RETR => Ok(Self::Retr { rs }),
                    _ => unreachable!(),
                }
            }
            Self::BCC => {
                let B { imm, func } = i.b();
                Ok(Self::B {
                    cc: BranchCond::new(func)?,
                    imm,
                })
            }
            // Stack Operations
            opcode @ (Self::PUSH | Self::POP) => {
                let M { rs, rd, .. } = i.m();
                match opcode {
                    Self::PUSH => Ok(Self::Push { rs }),
                    Self::POP => Ok(Self::Pop { rd }),
                    _ => unreachable!(),
                }
            }
            Self::ENTER => Ok(Self::Enter),
            Self::LEAVE => Ok(Self::Leave),
            // Data Flow
            Self::LI => {
                let F { imm, func, rd } = i.f();
                Ok(Self::Li {
                    rd,
                    func: LiType::new(func)?,
                    imm,
                })
            }
            opcode @ Self::LW..=Self::SB => {
                let E {
                    imm: off,
                    func: sh,
                    rs2: rn,
                    rs1: rs,
                    rd,
                } = i.e();
                match opcode {
                    Self::LW => Ok(Self::Lw { rd, rs, rn, sh, off }),
                    Self::LH => Ok(Self::Lh { rd, rs, rn, sh, off }),
                    Self::LHS => Ok(Self::Lhs { rd, rs, rn, sh, off }),
                    Self::LQ => Ok(Self::Lq { rd, rs, rn, sh, off }),
                    Self::LQS => Ok(Self::Lqs { rd, rs, rn, sh, off }),
                    Self::LB => Ok(Self::Lb { rd, rs, rn, sh, off }),
                    Self::LBS => Ok(Self::Lbs { rd, rs, rn, sh, off }),
                    Self::SW => Ok(Self::Sw { rd, rs, rn, sh, off }),
                    Self::SH => Ok(Self::Sh { rd, rs, rn, sh, off }),
                    Self::SQ => Ok(Self::Sq { rd, rs, rn, sh, off }),
                    Self::SB => Ok(Self::Sb { rd, rs, rn, sh, off }),
                    _ => unreachable!(),
                }
            }
            // Comparisons
            Self::CMPR => {
                let M { rs, rd, .. } = i.m();
                Ok(Self::Cmpr { r1: rd, r2: rs })
            }
            Self::CMPI => {
                let F { imm, func, rd } = i.f();
                Ok(Self::Cmpi {
                    r1: rd,
                    s: func.try_into_bool().ok_or(Interrupt::InvalidInstruction)?,
                    imm,
                })
            }
            // Arithmetic Operations
            // r-types
            opcode @ Self::ADDR..=Self::MODI if opcode % 2 == 0 => {
                let R { rs2: r2, rs1: r1, rd, .. } = i.r();
                match opcode {
                    Self::ADDR => Ok(Self::Addr { rd, r1, r2 }),
                    Self::SUBR => Ok(Self::Subr { rd, r1, r2 }),
                    Self::IMULR => Ok(Self::Imulr { rd, r1, r2 }),
                    Self::IDIVR => Ok(Self::Idivr { rd, r1, r2 }),
                    Self::UMULR => Ok(Self::Umulr { rd, r1, r2 }),
                    Self::UDIVR => Ok(Self::Udivr { rd, r1, r2 }),
                    Self::REMR => Ok(Self::Remr { rd, r1, r2 }),
                    Self::MODR => Ok(Self::Modr { rd, r1, r2 }),
                    _ => unreachable!(),
                }
            }
            // i-types
            opcode @ Self::ADDR..=Self::MODI => {
                let M { imm, rs: r1, rd } = i.m();
                match opcode {
                    Self::ADDI => Ok(Self::Addi { rd, r1, imm }),
                    Self::SUBI => Ok(Self::Subi { rd, r1, imm }),
                    Self::IMULI => Ok(Self::Imuli { rd, r1, imm }),
                    Self::IDIVI => Ok(Self::Idivi { rd, r1, imm }),
                    Self::UMULI => Ok(Self::Umuli { rd, r1, imm }),
                    Self::UDIVI => Ok(Self::Udivi { rd, r1, imm }),
                    Self::REMI => Ok(Self::Remi { rd, r1, imm }),
                    Self::MODI => Ok(Self::Modi { rd, r1, imm }),
                    _ => unreachable!(),
                }
            }

            // Bitwise Operations
            // r-types
            opcode @ Self::ANDR..=Self::BITI if opcode % 2 == 0 => {
                let R { rs2: r2, rs1: r1, rd, .. } = i.r();
                match opcode {
                    Self::ANDR => Ok(Self::Andr { rd, r1, r2 }),
                    Self::ORR => Ok(Self::Orr { rd, r1, r2 }),
                    Self::NORR => Ok(Self::Norr { rd, r1, r2 }),
                    Self::XORR => Ok(Self::Xorr { rd, r1, r2 }),
                    Self::SHLR => Ok(Self::Shlr { rd, r1, r2 }),
                    Self::ASRR => Ok(Self::Asrr { rd, r1, r2 }),
                    Self::LSRR => Ok(Self::Lsrr { rd, r1, r2 }),
                    Self::BITR => Ok(Self::Bitr { rd, r1, r2 }),
                    _ => unreachable!(),
                }
            }
            // i-types
            opcode @ Self::ANDR..=Self::BITI => {
                let M { imm, rs: r1, rd } = i.m();
                match opcode {
                    Self::ANDI => Ok(Self::Andi { rd, r1, imm }),
                    Self::ORI => Ok(Self::Ori { rd, r1, imm }),
                    Self::NORI => Ok(Self::Nori { rd, r1, imm }),
                    Self::XORI => Ok(Self::Xori { rd, r1, imm }),
                    Self::SHLI => Ok(Self::Shli { rd, r1, imm }),
                    Self::ASRI => Ok(Self::Asri { rd, r1, imm }),
                    Self::LSRI => Ok(Self::Lsri { rd, r1, imm }),
                    Self::BITI => Ok(Self::Biti { rd, r1, imm }),
                    _ => unreachable!(),
                }
            }
            // Floating-Point Operations
            opcode @ Self::FCMP..=Self::FNAN => {
                let E {
                    func, rs2: r2, rs1: r1, rd, ..
                } = i.e();
                let p = FloatPrecision::new(func)?;
                match opcode {
                    Self::FCMP => Ok(Self::Fcmp { r1, r2, p }),
                    Self::FTO => Ok(Self::Fto { rd, rs: r1, p }),
                    Self::FFROM => Ok(Self::Ffrom { rd, rs: r1, p }),
                    Self::FNEG => Ok(Self::Fneg { rd, rs: r1, p }),
                    Self::FABS => Ok(Self::Fabs { rd, rs: r1, p }),
                    Self::FADD => Ok(Self::Fadd { rd, r1, r2, p }),
                    Self::FSUB => Ok(Self::Fsub { rd, r1, r2, p }),
                    Self::FMUL => Ok(Self::Fmul { rd, r1, r2, p }),
                    Self::FDIV => Ok(Self::Fdiv { rd, r1, r2, p }),
                    Self::FMA => Ok(Self::Fma { rd, r1, r2, p }),
                    Self::FSQRT => Ok(Self::Fsqrt { rd, r1, p }),
                    Self::FMIN => Ok(Self::Fmin { rd, r1, r2, p }),
                    Self::FMAX => Ok(Self::Fmax { rd, r1, r2, p }),
                    Self::FSAT => Ok(Self::Fsat { rd, r1, p }),
                    Self::FCNV => Ok(Self::Fcnv {
                        rd,
                        r1,
                        p: FloatCastType::new(func)?,
                    }),
                    Self::FNAN => Ok(Self::Fnan { rd, r1, p }),
                    _ => unreachable!(),
                }
            }
            _ => Err(Interrupt::InvalidInstruction),
        }
    }
}
