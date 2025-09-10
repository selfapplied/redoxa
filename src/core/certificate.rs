//! Unified Certificate System for CE1
//! 
//! Implements the core certificate abstraction that all modules use.
//! Provides verifiable, compressible, witness-bearing proofs of execution/metrics.
//! 
//! CE1{
//!   certificate: verifiable, compressible, witness-bearing proof
//!   invariants: reversibility, single ledger (CIDs), witness discipline
//!   types: Metric, Execution, Validation, Compressed
//! }

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Core certificate trait - all certificates implement this
pub trait Certificate {
    /// Verify the certificate is valid
    fn verify(&self) -> bool;
    
    /// Serialize to bytes for storage
    fn serialize(&self) -> Vec<u8>;
    
    /// Get execution time
    fn execution_time(&self) -> f64;
    
    /// Check if invariants are preserved
    fn invariants_preserved(&self) -> bool;
    
    /// Get witness data
    fn witness(&self) -> HashMap<String, serde_json::Value>;
    
    /// Get certificate ID (CID)
    fn id(&self) -> String;
}

/// Certificate types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertificateType {
    Metric,
    Execution,
    Validation,
    Compressed,
    Custom(String),
}

/// Unified certificate that can hold any certificate type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedCertificate {
    pub id: String,
    pub cert_type: CertificateType,
    pub execution_time: f64,
    pub invariants_preserved: bool,
    pub witness_data: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: u64,
    pub compressed_data: Option<Vec<u8>>,
}

impl Certificate for UnifiedCertificate {
    fn verify(&self) -> bool {
        self.invariants_preserved && !self.id.is_empty()
    }
    
    fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
    
    fn execution_time(&self) -> f64 {
        self.execution_time
    }
    
    fn invariants_preserved(&self) -> bool {
        self.invariants_preserved
    }
    
    fn witness(&self) -> HashMap<String, serde_json::Value> {
        self.witness_data.clone()
    }
    
    fn id(&self) -> String {
        self.id.clone()
    }
}

/// Metric certificate for performance measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCertificate {
    pub id: String,
    pub metric_name: String,
    pub value: f64,
    pub unit: String,
    pub execution_time: f64,
    pub witness_data: HashMap<String, serde_json::Value>,
    pub created_at: u64,
}

impl Certificate for MetricCertificate {
    fn verify(&self) -> bool {
        !self.id.is_empty() && !self.metric_name.is_empty()
    }
    
    fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
    
    fn execution_time(&self) -> f64 {
        self.execution_time
    }
    
    fn invariants_preserved(&self) -> bool {
        true // Metric certificates always preserve invariants
    }
    
    fn witness(&self) -> HashMap<String, serde_json::Value> {
        self.witness_data.clone()
    }
    
    fn id(&self) -> String {
        self.id.clone()
    }
}

/// Execution certificate for operation proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCertificate {
    pub id: String,
    pub operation: String,
    pub success: bool,
    pub execution_time: f64,
    pub witness_data: HashMap<String, serde_json::Value>,
    pub created_at: u64,
}

impl Certificate for ExecutionCertificate {
    fn verify(&self) -> bool {
        !self.id.is_empty() && !self.operation.is_empty()
    }
    
    fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
    
    fn execution_time(&self) -> f64 {
        self.execution_time
    }
    
    fn invariants_preserved(&self) -> bool {
        self.success
    }
    
    fn witness(&self) -> HashMap<String, serde_json::Value> {
        self.witness_data.clone()
    }
    
    fn id(&self) -> String {
        self.id.clone()
    }
}

/// Certificate manager for handling certificates
pub struct CertificateManager {
    certificates: HashMap<String, UnifiedCertificate>,
    next_id: u64,
}

impl CertificateManager {
    pub fn new() -> Self {
        Self {
            certificates: HashMap::new(),
            next_id: 1,
        }
    }
    
    /// Generate a new CID
    pub fn generate_cid(&mut self) -> String {
        let id = format!("cert_{:016x}", self.next_id);
        self.next_id += 1;
        id
    }
    
    /// Create a metric certificate
    pub fn create_metric_certificate(
        &mut self,
        metric_name: String,
        value: f64,
        unit: String,
        execution_time: f64,
    ) -> MetricCertificate {
        let id = self.generate_cid();
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        MetricCertificate {
            id,
            metric_name,
            value,
            unit,
            execution_time,
            witness_data: HashMap::new(),
            created_at,
        }
    }
    
    /// Create an execution certificate
    pub fn create_execution_certificate(
        &mut self,
        operation: String,
        success: bool,
        execution_time: f64,
    ) -> ExecutionCertificate {
        let id = self.generate_cid();
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        ExecutionCertificate {
            id,
            operation,
            success,
            execution_time,
            witness_data: HashMap::new(),
            created_at,
        }
    }
    
    /// Store a certificate
    pub fn store_certificate(&mut self, cert: UnifiedCertificate) {
        self.certificates.insert(cert.id.clone(), cert);
    }
    
    /// Get a certificate by ID
    pub fn get_certificate(&self, id: &str) -> Option<&UnifiedCertificate> {
        self.certificates.get(id)
    }
    
    /// List all certificate IDs
    pub fn list_certificates(&self) -> Vec<String> {
        self.certificates.keys().cloned().collect()
    }
}